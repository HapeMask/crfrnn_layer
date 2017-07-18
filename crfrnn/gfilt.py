import os

import numpy as np
import theano
from theano.compile import optdb
from theano.tensor.basic import as_tensor_variable, get_scalar_constant_value
from theano.scalar.basic import constant
from theano.gof.op import OpenMPOp
from theano.gof.graph import Apply
from theano.gradient import DisconnectedType
from theano.gof.opt import local_optimizer, TopoOptimizer

from .phash import PermutohedralHashTable, MIN_QUAD_PROBES

try:
    import theano.gpuarray
    from theano.gpuarray.opt import register_opt, register_opt2, op_lifter
    from .gpugfilt import GpuGaussianFilter
    has_gpua = True
except:
    has_gpua = False


class GaussianFilter(OpenMPOp):
    __props__ = ("inplace",)

    def __init__(self, inplace=False, openmp=None):
        super(GaussianFilter, self).__init__(openmp=openmp)

        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [1]}

    def make_node(self, ref, values, ref_dim, val_dim, *_hash):
        assert(values.ndim == 3)
        ref = as_tensor_variable(ref.astype("float32"))
        values = as_tensor_variable(values.astype("float32"))

        ref_dim = get_scalar_constant_value(ref_dim)
        val_dim = get_scalar_constant_value(val_dim)
        if "int" not in str(ref_dim.dtype) or "int" not in str(val_dim.dtype):
            raise ValueError("ref_dim and val_dim must be integers.")

        scaled_ref = ref * float(np.sqrt(2/3) * (ref_dim+1))

        if len(_hash) == 0:
            hash_struct = PermutohedralHashTable()(scaled_ref, ref_dim)
        else:
            assert(len(_hash) == 6)
            hash_struct = [as_tensor_variable(v) for v in _hash]

        # Should we not do this?
        bcast = [False for _ in range(3)]
        if val_dim == 1:
            bcast[0] = True

        out_type = values.type.clone(broadcastable=bcast)

        ref_dim = constant(ref_dim, dtype="int32", name="ref_dim")
        val_dim = constant(val_dim, dtype="int32", name="val_dim")

        inputs = [ref, values, ref_dim, val_dim] + hash_struct
        return Apply(self, inputs, [out_type()])

    def infer_shape(self, node, in_shapes):
        return [in_shapes[1]]

    def connection_pattern(self, node):
        cp = [[False] for _ in range(len(node.inputs))]
        cp[0][0] = True
        cp[1][0] = True
        return cp

    def grad(self, inputs, ograds):
        ref, values, ref_dim, val_dim = inputs[:4]
        hash_struct = inputs[4:]
        ograd = ograds[0]

        ref_dim = get_scalar_constant_value(ref_dim)
        val_dim = get_scalar_constant_value(val_dim)

        def _conv(x):
            return GaussianFilter()(ref, x, ref_dim, val_dim, *hash_struct)

        # Since the kernels are separable and symmetric, the gradient w.r.t.
        # input is just the same filtering applied to the output grads.
        grad_i = _conv(ograd)

        def _gradr(r_i, vals, og, *args):
            return (og * (_conv(vals*r_i) - r_i*_conv(vals)) +
                    vals * (_conv(og*r_i) - r_i*_conv(og)))

        grad_r, _ = theano.scan(fn=_gradr, sequences=[ref],
                                non_sequences=[values, ograd] + hash_struct,
                                outputs_info=None)

        grad_r = grad_r.sum(axis=1, acc_dtype="float32")

        grads = [DisconnectedType()() for i in range(len(inputs))]
        grads[0] = grad_r
        grads[1] = grad_i
        return grads

    def _macros(self, node, name):
        define_template = "#define %s %s\n"
        undef_template = "#undef %s\n"
        define_macros = []
        undef_macros = []

        rdim = get_scalar_constant_value(node.inputs[2])
        vdim = get_scalar_constant_value(node.inputs[3])

        define_macros.append(define_template % ("DIM_SPECIFIC(str)",
                                                "str##_%d_%d" % (rdim, vdim)))
        undef_macros.append(undef_template % "DIM_SPECIFIC")

        consts = {"REF_DIM":            str(rdim),
                  "VAL_DIM":            str(vdim),
                  "KEY_DIM":            str(rdim),
                  "MIN_QUAD_PROBES":    str(MIN_QUAD_PROBES),
                  "GID_0":              "filt_fakegpu_GID_0",
                  "LID_0":              "filt_fakegpu_LID_0",
                  "LDIM_0":             "filt_fakegpu_LDIM_0",
                  "KERNEL":             "",
                  "GLOBAL_MEM":         ""}

        for k, v in consts.items():
            define_macros.append(define_template % (k, v))
            undef_macros.append(undef_template % k)

        return ''.join(define_macros), ''.join(undef_macros)

    def c_support_code(self):
        code = """
int filt_fakegpu_GID_0;
#pragma omp threadprivate(filt_fakegpu_GID_0)

const int filt_fakegpu_LID_0 = 0;
const int filt_fakegpu_LDIM_0 = 1;
bool valid_output_ptr(PyArrayObject* o,
                      int typecode,
                      npy_intp exp_ndim,
                      const npy_intp* exp_shape) {
if (o == NULL || PyArray_NDIM(o) != exp_ndim || PyArray_TYPE(o) != typecode) {
    return false;
}
for(int i=0; i<exp_ndim; ++i) {
    if(PyArray_DIMS(o)[i] != exp_shape[i]) { return false; }
}
return true;
}"""
        return code

    def c_support_code_apply(self, node, name):
        def_macros, undef_macros = self._macros(node, name)
        hsup = (PermutohedralHashTable._hash_support_code() + "\n" + \
                PermutohedralHashTable._lookup_code())

        knames = ["splat", "blur", "slice"]
        kcodes = ["".join(open("%s%s%s.cu" %
                  (os.path.dirname(__file__), os.path.sep, kn)).readlines())
                  for kn in knames]
        code = "\n".join(kcodes)
        code = "\n".join([def_macros, hsup, code, undef_macros])
        code = code.replace("__inline__", "inline")
        code = code.replace("__device__", "")
        return code

    def c_code(self, node, name, inputs, outputs, sub):
        values = inputs[1]
        entries, keys, neib_ents, barycentric, valid_entries, nv = inputs[4:]
        output = outputs[0]

        rdim = get_scalar_constant_value(node.inputs[2])
        vdim = get_scalar_constant_value(node.inputs[3])

        fail = sub["fail"]
        inplace = "1" if self.inplace else "0"

        code = """
npy_intp val_dims[3];
npy_intp tmp_val_dims[2];
npy_intp output_dims[3];

val_dims[0] = PyArray_DIMS(%(values)s)[0];
val_dims[1] = PyArray_DIMS(%(values)s)[1];
val_dims[2] = PyArray_DIMS(%(values)s)[2];

size_t N = val_dims[1] * val_dims[2];
size_t cap = N*(%(rdim)s+1);

size_t ls_N, gs_N, ls_valid, gs_valid;
int nv = *((int*)PyArray_DATA(%(nv)s));

PyArrayObject* tmp_vals_1 = NULL;
PyArrayObject* tmp_vals_2 = NULL;
PyArrayObject* tmp_vptr_1 = NULL;
PyArrayObject* tmp_vptr_2 = NULL;
PyArrayObject* tmp_swap = NULL;

PyArrayObject* vcontig = NULL;
bool should_decref_vcontig = false;

if(val_dims[0] != %(vdim)s) {
    PyErr_Format(PyExc_ValueError,
        "blur error: bad input shape 0.\\nExpected %(vdim)s, got %%d",
        val_dims[0]);
    %(fail)s;
}

if(val_dims[1] != PyArray_DIMS(%(barycentric)s)[1] ||
   val_dims[2] != PyArray_DIMS(%(barycentric)s)[2]) {
    PyErr_Format(PyExc_ValueError,
            "blur error: bad input h/w.\\nExpected (%%d, %%d), got (%%d, %%d)",
            val_dims[1], val_dims[2]);
    %(fail)s;
}

tmp_val_dims[0] = cap;
tmp_val_dims[1] = val_dims[0];

output_dims[0] = val_dims[0];
output_dims[1] = val_dims[1];
output_dims[2] = val_dims[2];

tmp_vals_1 = (PyArrayObject*)PyArray_ZEROS(2, tmp_val_dims, NPY_FLOAT, 0);
tmp_vals_2 = (PyArrayObject*)PyArray_ZEROS(2, tmp_val_dims, NPY_FLOAT, 0);
if (!tmp_vals_1 || !tmp_vals_2) {
    PyErr_Format(PyExc_RuntimeError,
                 "error allocating temporary filtering storage.");
    %(fail)s;
}

tmp_vptr_1 = tmp_vals_1;
tmp_vptr_2 = tmp_vals_2;

if(%(inplace)s) {
    Py_XDECREF(%(output)s);
    %(output)s = %(values)s;
    Py_INCREF(%(output)s);
} else if(!valid_output_ptr(%(output)s, NPY_FLOAT, 3, output_dims)) {
    Py_XDECREF(%(output)s);
    %(output)s = (PyArrayObject*)PyArray_ZEROS(3, output_dims, NPY_FLOAT, 0);
}

if (!%(output)s) {
    PyErr_Format(PyExc_MemoryError,
        "error performing gaussian blur: failed to allocate output storage.");
    %(fail)s;
}

if (!PyArray_IS_C_CONTIGUOUS(%(values)s)) {
    should_decref_vcontig = true;
}
vcontig = PyArray_GETCONTIGUOUS(%(values)s);

#pragma omp parallel for
for(int i=0; i<N; ++i) {
    filt_fakegpu_GID_0 = i;
    splat_%(rdim)s_%(vdim)s(
        (float*)PyArray_DATA(vcontig), 0,
        (float*)PyArray_DATA(%(barycentric)s), 0,
        (int*)PyArray_DATA(%(entries)s), 0,
        (int*)PyArray_DATA(%(neib_ents)s), 0,
        (float*)PyArray_DATA(tmp_vals_1),
        N);
}

for(int ax=0; ax<%(rdim)s+1; ++ax) {
    #pragma omp parallel for
    for(int i=0; i<nv; ++i) {
        filt_fakegpu_GID_0 = i;
        blur_%(rdim)s_%(vdim)s(
            (float*)PyArray_DATA(tmp_vptr_2),
            (int*)PyArray_DATA(%(entries)s), 0,
            (int*)PyArray_DATA(%(valid_entries)s), 0,
            (short*)PyArray_DATA(%(keys)s), 0,
            (float*)PyArray_DATA(tmp_vptr_1),
            cap, nv, ax);
    }

    tmp_swap = tmp_vptr_1;
    tmp_vptr_1 = tmp_vptr_2;
    tmp_vptr_2 = tmp_swap;
}

#pragma omp parallel for
for(int i=0; i<N; ++i) {
    filt_fakegpu_GID_0 = i;
    slice_%(rdim)s_%(vdim)s(
        (float*)PyArray_DATA(%(output)s), 0,
        (float*)PyArray_DATA(%(barycentric)s), 0,
        (int*)PyArray_DATA(%(entries)s), 0,
        (int*)PyArray_DATA(%(neib_ents)s), 0,
        (float*)PyArray_DATA(tmp_vptr_2),
        N);
}

if (should_decref_vcontig) {
    Py_DECREF(vcontig);
}
"""
        return code % locals()


def register_inplace(*tags, **kwargs):
    def f(local_opt):
        name = (kwargs and kwargs.pop('name')) or local_opt.__name__
        optdb.register(
            name, TopoOptimizer(
                local_opt, failure_callback=TopoOptimizer.warn_inplace),
            60, 'fast_run', 'inplace', *tags)
        return local_opt
    return f

@register_inplace()
@local_optimizer([GaussianFilter], inplace=True)
def local_gaussian_filter_inplace(node):
    if isinstance(node.op, GaussianFilter) and not node.op.inplace:
        return [GaussianFilter(inplace=True,
                               openmp=node.op.openmp)(*node.inputs)]

if has_gpua:
    @register_opt('fast_compile')
    @op_lifter([GaussianFilter])
    @register_opt2([GaussianFilter], 'fast_compile')
    def local_gaussian_filter_to_gpu(op, context_name, inputs, outputs):
        return GpuGaussianFilter(inplace=op.inplace)(*inputs)

def gaussian_filter(ref_img, values, kern_std, ref_dim=None, val_dim=None,
                    *_hash):
    """Applies a high-dimensional Gaussian filter to 'values' with pairwise
    Gaussian weights based on features in 'ref_img'.

    Parameters
    ----------

    ref_img : array_like, shape (ref_dim, H, W)
        The reference image from which to derive the pairwise Gaussian weights
        (the locations for each image pixel in a high-dimensional space).

    values : array_like, shape (val_dim, H, W)
        The image we are going to filter.

    kern_std : array_like, shape (ref_dim, )
        Standard deviation of the Gaussian filter in each dimension.

    ref_dim : int or None
        The reference image dimensionality. Must be a known scalar constant.
        For a color bilateral filter, this is 5: x, y, r, g, b.

        If None, attempt to infer the dimensionality from the shape of
        'ref_img'.

    val_dim : int or None
        The image dimensionality (color channels, usually). Must be a known
        scalar constant.

        If None, attempt to infer the dimensionality from the shape of
        'values'.
"""

    if ref_dim is None:
        ref_dim = get_scalar_constant_value(ref_img.shape[0])

    if val_dim is None:
        val_dim = get_scalar_constant_value(values.shape[0])

    scaled_ref = ref_img / kern_std[:, np.newaxis, np.newaxis]
    return GaussianFilter()(scaled_ref, values, ref_dim, val_dim, *_hash)
