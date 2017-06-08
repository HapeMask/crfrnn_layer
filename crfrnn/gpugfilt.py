import os

import numpy as np
import theano
from pygpu.gpuarray import GpuArray, SIZE
from theano.tensor.basic import as_tensor_variable, get_scalar_constant_value
from theano.scalar.basic import constant
from theano.gpuarray.type import get_context
from theano.gpuarray.basic_ops import GpuKernelBase, gpu_contiguous, Kernel
from theano.gof.op import Op
from theano.gof.graph import Apply
from theano.gradient import DisconnectedType
from theano.gof.opt import local_optimizer
from theano.gpuarray.opt import register_inplace

from .gpuhash import GpuHashTable, MIN_QUAD_PROBES


class GpuGaussianFilter(GpuKernelBase, Op):
    __props__ = ("inplace", "context_name")

    def __init__(self, inplace=False, context_name=None):
        self.inplace = inplace
        self.context_name = context_name
        if self.inplace:
            self.destroy_map = {0: [1]}

    def get_params(self, node):
        return get_context(self.context_name)

    def make_node(self, ref, values, ref_dim, val_dim, *_hash):
        assert(values.ndim == 3)
        ref = gpu_contiguous(as_tensor_variable(ref.astype("float32")))
        values = gpu_contiguous(as_tensor_variable(values.astype("float32")))

        ref_dim = get_scalar_constant_value(ref_dim)
        val_dim = get_scalar_constant_value(val_dim)
        if "int" not in str(ref_dim.dtype) or "int" not in str(val_dim.dtype):
            raise ValueError("ref_dim and val_dim must be integers.")

        scaled_ref = ref * float(np.sqrt(2/3) * (ref_dim+1))

        if len(_hash) == 0:
            hash_struct = GpuHashTable()(scaled_ref, ref_dim)
        else:
            assert(len(_hash) == 6)
            tv = [as_tensor_variable(v) for v in _hash]
            hash_struct = [gpu_contiguous(v) for v in tv]

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
            return GpuGaussianFilter()(ref, x, ref_dim, val_dim, *hash_struct)

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
                  "MIN_QUAD_PROBES":    str(MIN_QUAD_PROBES)}

        for k, v in consts.items():
            define_macros.append(define_template % (k, v))
            undef_macros.append(undef_template % k)

        return ''.join(define_macros), ''.join(undef_macros)

    def gpu_kernels(self, node, name):
        rdim = get_scalar_constant_value(node.inputs[2])
        vdim = get_scalar_constant_value(node.inputs[3])

        flags = Kernel.get_flags(node.inputs[0].dtype, node.inputs[1].dtype)

        def_macros, undef_macros = self._macros(node, name)
        hsup = (GpuHashTable._hash_support_code() + "\n" +
                GpuHashTable._lookup_code())

        knames = ["splat", "blur", "slice"]
        kcodes = ["".join(open("%s%s%s.cu" %
                  (os.path.dirname(__file__), os.path.sep, kn)).readlines())
                  for kn in knames]
        kcodes = ["\n".join([def_macros, hsup, code, undef_macros])
                  for code in kcodes]
        kparams = ([GpuArray, SIZE,
                    GpuArray, SIZE,
                    GpuArray, SIZE,
                    GpuArray, SIZE,
                    GpuArray, SIZE],

                   [GpuArray,
                    GpuArray, SIZE,
                    GpuArray, SIZE,
                    GpuArray, SIZE,
                    GpuArray,
                    SIZE, SIZE, SIZE],

                   [GpuArray, SIZE,
                    GpuArray, SIZE,
                    GpuArray, SIZE,
                    GpuArray, SIZE,
                    GpuArray, SIZE])

        return [Kernel(code=kcode, name="%s_%d_%d" % (kname, rdim, vdim),
                params=kparams, flags=flags)
                for kcode, kname, kparams in zip(kcodes, knames, kparams)]

    def c_header_dirs(self):
        gpua_dir = os.path.dirname(theano.gpuarray.__file__)
        return [gpua_dir]

    def c_headers(self):
        return ["<gpuarray/types.h>",
                "<gpuarray/kernel.h>",
                "<numpy_compat.h>",
                "<gpuarray_helper.h>"]

    def c_support_code(self):
        code = """
bool valid_output_ptr(const PyGpuArrayObject* o,
                      int typecode,
                      size_t exp_ndim,
                      const size_t* exp_shape) {
if (o == NULL || o->ga.nd != exp_ndim || o->ga.typecode != typecode) {
    return false;
}
for(int i=0; i<exp_ndim; ++i) {
    if(PyGpuArray_DIMS(o)[i] != exp_shape[i]) { return false; }
}
return true;
}"""
        return code

    def c_code(self, node, name, inputs, outputs, sub):
        values = inputs[1]
        entries, keys, neib_ents, barycentric, valid_entries, nv = inputs[4:]
        output = outputs[0]

        rdim = get_scalar_constant_value(node.inputs[2])
        vdim = get_scalar_constant_value(node.inputs[3])

        fail = sub["fail"]
        ctx = sub["params"]
        kname_splat = "k_splat_%d_%d" % (rdim, vdim)
        kname_blur = "k_blur_%d_%d" % (rdim, vdim)
        kname_slice = "k_slice_%d_%d" % (rdim, vdim)
        inplace = "1" if self.inplace else "0"

        code = """
int err = GA_NO_ERROR;

size_t val_dims[3];
size_t tmp_val_dims[2];
size_t output_dims[3];

val_dims[0] = PyGpuArray_DIMS(%(values)s)[0];
val_dims[1] = PyGpuArray_DIMS(%(values)s)[1];
val_dims[2] = PyGpuArray_DIMS(%(values)s)[2];

size_t N = val_dims[1] * val_dims[2];
size_t cap = N*(%(rdim)s+1);

size_t ls_N, gs_N, ls_valid, gs_valid;
int nv;
GpuArray_read((void*)(&nv), sizeof(int), &%(nv)s->ga);

GpuArray tmp_vals_1, tmp_vals_2;
GpuArray* tmp_vptr_1 = &tmp_vals_1;
GpuArray* tmp_vptr_2 = &tmp_vals_2;
GpuArray* tmp_swap = NULL;

if(val_dims[0] != %(vdim)s) {
    PyErr_Format(PyExc_ValueError,
        "blur error: bad input shape 0.\\nExpected %(vdim)s, got %%d",
        val_dims[0]);
    %(fail)s;
}

if(val_dims[1] != PyGpuArray_DIMS(%(barycentric)s)[1] ||
   val_dims[2] != PyGpuArray_DIMS(%(barycentric)s)[2]) {
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

if(%(inplace)s) {
    Py_XDECREF(%(output)s);
    %(output)s = %(values)s;
    Py_INCREF(%(output)s);
} else if(!valid_output_ptr(%(output)s, GA_FLOAT, 3, output_dims)) {
    Py_XDECREF(%(output)s);
    %(output)s = pygpu_zeros(3, output_dims, GA_FLOAT, GA_C_ORDER,
                             %(ctx)s, Py_None);
}

err = GpuArray_zeros(&tmp_vals_1, %(ctx)s->ctx, GA_FLOAT, 2, tmp_val_dims,
                     GA_C_ORDER);
if(err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
                 "gpuarray error allocating memory:\\n%%s.\\n",
                 GpuArray_error(&tmp_vals_1, err));
    %(fail)s;
}

err = GpuArray_zeros(&tmp_vals_2, %(ctx)s->ctx, GA_FLOAT, 2, tmp_val_dims,
                     GA_C_ORDER);
if(err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
                 "gpuarray error allocating memory:\\n%%s.\\n",
                 GpuArray_error(&tmp_vals_2, err));
    %(fail)s;
}

if (!%(output)s) {
    PyErr_Format(PyExc_MemoryError,
        "error performing gaussian blur: failed to allocate output storage.");
    %(fail)s;
}

gs_N = ls_N = 0;
GpuKernel_sched(&%(kname_splat)s, N, &gs_N, &ls_N);
gs_N = N / ls_N;
if (ls_N*gs_N < N) { ++gs_N; }

err = splat_%(rdim)s_%(vdim)s_call(1, &gs_N, &ls_N, 0,
    %(values)s->ga.data, %(values)s->ga.offset / sizeof(float),
    %(barycentric)s->ga.data, %(barycentric)s->ga.offset / sizeof(float),
    %(entries)s->ga.data, %(entries)s->ga.offset / sizeof(int),
    %(neib_ents)s->ga.data, %(neib_ents)s->ga.offset / sizeof(int),
    tmp_vals_1.data,
    N);

if(err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError, "gpuarray error splatting:\\n%%s.\\n",
        GpuKernel_error(&%(kname_splat)s, err));
    %(fail)s;
}
GpuArray_sync(&tmp_vals_1);

gs_valid = ls_valid = 0;
GpuKernel_sched(&%(kname_blur)s, nv, &gs_valid, &ls_valid);
gs_valid = nv / ls_valid;
if (ls_valid*gs_valid < nv) { ++gs_valid; }

for(int ax=0; ax<%(rdim)s+1; ++ax) {
    err = blur_%(rdim)s_%(vdim)s_call(1, &gs_valid, &ls_valid, 0,
        tmp_vptr_2->data,
        %(entries)s->ga.data, %(entries)s->ga.offset / sizeof(int),
        %(valid_entries)s->ga.data, %(valid_entries)s->ga.offset / sizeof(int),
        %(keys)s->ga.data, %(keys)s->ga.offset / sizeof(short),
        tmp_vptr_1->data,
        cap, nv, ax);

    if(err != GA_NO_ERROR) {
        PyErr_Format(PyExc_RuntimeError, "gpuarray error blurring:\\n%%s.\\n",
            GpuKernel_error(&%(kname_blur)s, err));
        %(fail)s;
    }

    GpuArray_sync(tmp_vptr_2);

    tmp_swap = tmp_vptr_1;
    tmp_vptr_1 = tmp_vptr_2;
    tmp_vptr_2 = tmp_swap;
}

err = slice_%(rdim)s_%(vdim)s_call(1, &gs_N, &ls_N, 0,
    %(output)s->ga.data, %(output)s->ga.offset / sizeof(float),
    %(barycentric)s->ga.data, %(barycentric)s->ga.offset / sizeof(float),
    %(entries)s->ga.data, %(entries)s->ga.offset / sizeof(int),
    %(neib_ents)s->ga.data, %(neib_ents)s->ga.offset / sizeof(int),
    tmp_vptr_2->data,
    N);

if(err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError, "gpuarray error slicing:\\n%%s.\\n",
        GpuKernel_error(&%(kname_slice)s, err));
    %(fail)s;
}

GpuArray_sync(&%(output)s->ga);
GpuArray_clear(&tmp_vals_1);
GpuArray_clear(&tmp_vals_2);
"""
        return code % locals()


@register_inplace()
@local_optimizer([GpuGaussianFilter], inplace=True)
def local_gaussian_filter_inplace(node):
    if isinstance(node.op, GpuGaussianFilter) and not node.op.inplace:
        return [GpuGaussianFilter(inplace=True,
                                  context_name=node.op.context_name
                                  )(*node.inputs)]


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
    return GpuGaussianFilter()(scaled_ref, values, ref_dim, val_dim, *_hash)
