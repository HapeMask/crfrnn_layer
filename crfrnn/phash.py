import os

import numpy as np
import theano
from theano.tensor import TensorType
from theano.tensor.basic import as_tensor_variable, get_scalar_constant_value
from theano.scalar.basic import constant
from theano.gof.op import OpenMPOp
from theano.gof.graph import Apply
from theano.gof.null_type import NullType

try:
    import theano.gpuarray
    from theano.gpuarray.opt import register_opt, register_opt2, op_lifter
    from .gpuhash import GpuHashTable
    has_gpua = True
except:
    has_gpua = False

MIN_QUAD_PROBES = 10000


class PermutohedralHashTable(OpenMPOp):
    """Represents the construction of the hash table / slicing table for a
    permutohedral lattice built from a reference image.

    Outputs
    -----
    entries : tensor, shape = [table_capacity]
        An array of indices mapping a given hash value to the corresponding
        location in one of the auxiliary arrays (keys, valid entries, etc...)

    keys : tensor, shape = [table_capacity, d]
        An array of keys for inserted items. Since the table corresponds to a
        permutohedral lattice, the keys are points on the d+1 dimensional
        hyperplane
        :math:`\mathbf{x}\cdot\mathbf{1} = 0,\mathbf{x} \in \mathbb{Z}^{d+1}`,
        and the last dimension (or any other single dimension) is redundant.

    neighbor_entries : tensor, shape = [d+1, H, W]
        An array of d+1 table indices for each each pixel mapping it to its d+1
        neighboring points on the lattice.

    barycentric_coords : tensor, shape = [d+1, H, W]
        Barycentric coordinates within the surrounding simplex for each pixel.

    valid_entries : tensor, shape = [table_capacity]
        An array of entries remaining after duplicates have been pruned (the
        hash table construction may insert the same key twice in different
        locations, only the first is used).

    n_valid : tensor, shape = [1]
        Number of non-duplicate entries in the table (the number of non-'-1'
        values in valid_entries).
    """

    def make_node(self, points, dim):
        assert(points.ndim == 3)
        points = as_tensor_variable(points.astype("float32"))

        dim = get_scalar_constant_value(dim)
        if "int" not in str(dim.dtype):
            raise ValueError("dim must be an integer.")

        dim = constant(dim, dtype="int32", name="dim")

        entries_type = TensorType("int32", broadcastable=(False,))
        keys_type = TensorType("int16", broadcastable=(False, False))
        neib_ent_type = TensorType("int32",
                                   broadcastable=(False, False, False))
        bary_type = TensorType("float32",
                               broadcastable=points.type.broadcastable)

        valid_entries_type = TensorType("int32", broadcastable=(False,))
        n_valid_type = TensorType("int32", broadcastable=(False,))

        out_vars = [entries_type(name="hash_entries"),
                    keys_type(name="hash_keys"),
                    neib_ent_type(name="neighbor_entries"),
                    bary_type(name="barycentric_coords"),
                    valid_entries_type(name="valid_entries"),
                    n_valid_type(name="n_valid")]

        # Two sets of entries can't be meaningfully compared without also
        # having the corresponding keys. Since we can only define per-output
        # comparisons, we have to hope that any time someone compares two
        # tables for equality, they will check all outputs.
        out_vars[0].tag.values_eq_approx = lambda e1, e2: True
        out_vars[2].tag.values_eq_approx = lambda e1, e2: True

        # The number of valid entries between two equivalent tables may be
        # different since it includes duplicates.
        out_vars[5].tag.values_eq_approx = lambda n1, n2: True

        def keys_comparison(k1, k2):
            k1 = [tuple(k) for k in np.asarray(k1)]
            k2 = [tuple(k) for k in np.asarray(k2)]
            return set(k1) == set(k2)
        out_vars[1].tag.values_eq_approx = keys_comparison

        def valid_entries_comparison(e1, e2):
            e1 = np.asarray(e1)
            e2 = np.asarray(e2)
            return len(np.unique(e1)) == len(np.unique(e2))
        out_vars[4].tag.values_eq_approx = valid_entries_comparison

        return Apply(self, [points, dim], out_vars)

    def infer_shape(self, node, in_shapes):
        dim = get_scalar_constant_value(node.inputs[1])
        point_shp = in_shapes[0]
        h, w = point_shp[:2]
        N = h*w
        cap = N*(dim+1)

        return [(cap,), (cap, dim), (dim+1, h, w), (dim+1, h, w), (cap,), (1,)]

    def grad(self, inputs, ograds):
        grads = [NullType("Tried to take gradient thru hashtable.")()
                 for i in range(len(inputs))]
        return grads

    def _macros(self, node, name):
        define_template = "#define %s %s\n"
        undef_template = "#undef %s\n"
        define_macros = []
        undef_macros = []

        dim = get_scalar_constant_value(node.inputs[1])
        define_macros.append(define_template % ("DIM_SPECIFIC(str)",
                                                "str##_%d" % dim))
        undef_macros.append(undef_template % "DIM_SPECIFIC")

        consts = {"REF_DIM":            str(dim),
                  "KEY_DIM":            str(dim),
                  "DR":                 "%s.f" % str(dim),
                  "INV_DR1":            "(1.f / (%s.f+1.f))" % str(dim),
                  "MIN_QUAD_PROBES":    str(MIN_QUAD_PROBES),
                  "GID_0":              "hash_fakegpu_GID_0",
                  "LID_0":              "hash_fakegpu_LID_0",
                  "LDIM_0":             "hash_fakegpu_LDIM_0",
                  "KERNEL":             "",
                  "GLOBAL_MEM":         ""}

        for k, v in consts.items():
            define_macros.append(define_template % (k, v))
            undef_macros.append(undef_template % k)

        return ''.join(define_macros), ''.join(undef_macros)

    @staticmethod
    def _hash_support_code():
        code = """
template <typename T>
T atomicAdd(T* addr, T val) {
    T old;
    #pragma omp critical
    {
        old = *addr;
        *addr = old + val;
    }
    return old;
}

template <typename T>
T atomicCAS(T* addr, T compare, T val) {
    T old;
    #pragma omp critical
    {
        old = *addr;
        *addr = (old == compare) ? val : old;
    }
    return old;
}

template <typename T>
inline T min(T a, T b) {
    return ((a > b) ? b : a);
}

template <typename T>
inline T max(T a, T b) {
    return ((a > b) ? a : b);
}

unsigned int DIM_SPECIFIC(hash)(const short* key) {
    unsigned int h = 0;
    for (int i=0; i < KEY_DIM; ++i) {
        h ^= ((unsigned int)key[i]) << ((31/KEY_DIM)*i);
    }
    return h;
}

bool DIM_SPECIFIC(key_cmp)(const short* key1, const short* key2) {
    for(int i=0; i<KEY_DIM; ++i) {
        if (key1[i] != key2[i]) { return false; }
    }
    return true;

}"""
        return code

    @staticmethod
    def _lookup_code():
        return "".join(open(
                    os.path.dirname(__file__) +
                    os.path.sep+"lookup.cu").readlines()
                    ).replace("__device__", "inline").replace(
                            "__inline__", "")

    def c_support_code(self):
        code = """
int hash_fakegpu_GID_0;
#pragma omp threadprivate(hash_fakegpu_GID_0)

const int hash_fakegpu_LID_0 = 0;
const int hash_fakegpu_LDIM_0 = 1;
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
        hsup = (self._hash_support_code() + "\n" + self._lookup_code())

        knames = ["build_hash", "dedup", "find_valid"]
        kcodes = ["".join(open("%s%s%s.cu" %
                  (os.path.dirname(__file__), os.path.sep, kn)).readlines())
                  for kn in knames]
        code = "\n".join(kcodes)
        code = "\n".join([def_macros, hsup, code, undef_macros])
        code = code.replace("__inline__", "inline")
        code = code.replace("__device__", "")
        return code

    def c_code(self, node, name, inputs, outputs, sub):
        points = inputs[0]
        entries, keys, neib_ents, barycentric, valid_entries, n_valid = outputs
        dim = get_scalar_constant_value(node.inputs[1])
        fail = sub["fail"]

        code = """
npy_intp point_dims[3];
npy_intp entries_dim[1];
npy_intp keys_dims[2];
npy_intp neib_ents_dims[3];
npy_intp barycentric_dims[3];
npy_intp valid_entries_dim[1];

npy_intp n_valid_dim[1];
n_valid_dim[0] = 1;

point_dims[0] = PyArray_DIMS(%(points)s)[0];
point_dims[1] = PyArray_DIMS(%(points)s)[1];
point_dims[2] = PyArray_DIMS(%(points)s)[2];

npy_intp N = point_dims[1] * point_dims[2];
npy_intp cap = N*(point_dims[0]+1);

PyArrayObject* pcontig = NULL;
bool should_decref_pcontig = false;

if(point_dims[0] != %(dim)s) {
    PyErr_Format(PyExc_ValueError,
        "hashtable error: incorrect input dim 0.\\nExpected %(dim)s got %%d",
        point_dims[0]);
    %(fail)s;
}

if(PyArray_TYPE(%(points)s) != NPY_FLOAT) {
    PyErr_Format(PyExc_ValueError,
        "hashtable error: incorrect dtype for points.");
    %(fail)s;
}

entries_dim[0] = cap;

keys_dims[0] = cap;
keys_dims[1] = point_dims[0];

neib_ents_dims[0] = point_dims[0]+1;
neib_ents_dims[1] = point_dims[1];
neib_ents_dims[2] = point_dims[2];

barycentric_dims[0] = point_dims[0]+1;
barycentric_dims[1] = point_dims[1];
barycentric_dims[2] = point_dims[2];

valid_entries_dim[0] = cap;

if(!valid_output_ptr(%(entries)s, NPY_INT, 1, entries_dim)) {
    Py_XDECREF(%(entries)s);
    %(entries)s = (PyArrayObject*)PyArray_EMPTY(1, entries_dim, NPY_INT, 0);
}
if(!valid_output_ptr(%(keys)s, NPY_SHORT, 2, keys_dims)) {
    Py_XDECREF(%(keys)s);
    %(keys)s = (PyArrayObject*)PyArray_ZEROS(2, keys_dims, NPY_SHORT, 0);
}
if(!valid_output_ptr(%(neib_ents)s, NPY_INT, 3, neib_ents_dims)) {
    Py_XDECREF(%(neib_ents)s);
    %(neib_ents)s = (PyArrayObject*)PyArray_ZEROS(3, neib_ents_dims, NPY_INT, 0);
}
if(!valid_output_ptr(%(barycentric)s, NPY_FLOAT, 3, barycentric_dims)) {
    Py_XDECREF(%(barycentric)s);
    %(barycentric)s = (PyArrayObject*)PyArray_ZEROS(3, barycentric_dims, NPY_FLOAT, 0);
}
if(!valid_output_ptr(%(valid_entries)s, NPY_INT, 1, valid_entries_dim)) {
    Py_XDECREF(%(valid_entries)s);
    %(valid_entries)s = (PyArrayObject*)PyArray_ZEROS(1, valid_entries_dim, NPY_INT, 0);
}
if(!valid_output_ptr(%(n_valid)s, NPY_INT, 1, n_valid_dim)) {
    Py_XDECREF(%(n_valid)s);
    %(n_valid)s = (PyArrayObject*)PyArray_ZEROS(1, n_valid_dim, NPY_INT, 0);
} else {
    PyArray_FillWithScalar(%(n_valid)s, PyLong_FromLong(0));
}

if (!(%(entries)s && %(keys)s && %(neib_ents)s && %(barycentric)s &&
    %(valid_entries)s)) {

    PyErr_Format(PyExc_MemoryError,
            "error building hash table: failed to allocate output storage.");
    %(fail)s;
}

if (!PyArray_IS_C_CONTIGUOUS(%(points)s)) {
    should_decref_pcontig = true;
}
pcontig = PyArray_GETCONTIGUOUS(%(points)s);

PyArray_FillWithScalar(%(entries)s, PyLong_FromLong(-1));

#pragma omp parallel for
for(int i=0; i<N; ++i) {
    hash_fakegpu_GID_0 = i;
    build_hash_%(dim)s(
        (float*)PyArray_DATA(%(points)s), 0,
        (int*)PyArray_DATA(%(entries)s), 0,
        (short*)PyArray_DATA(%(keys)s), 0,
        (int*)PyArray_DATA(%(neib_ents)s), 0,
        (float*)PyArray_DATA(%(barycentric)s), 0,
        cap, N);
}

#pragma omp parallel for
for(int i=0; i<cap; ++i) {
    hash_fakegpu_GID_0 = i;
    dedup_%(dim)s(
        (int*)PyArray_DATA(%(entries)s), 0,
        (short*)PyArray_DATA(%(keys)s), 0,
        cap);
}

#pragma omp parallel for
for(int i=0; i<cap; ++i) {
    hash_fakegpu_GID_0 = i;
    find_valid_%(dim)s(
        (int*)PyArray_DATA(%(entries)s), 0,
        (int*)PyArray_DATA(%(valid_entries)s), 0,
        (int*)PyArray_DATA(%(n_valid)s), 0,
        cap);
}

if (should_decref_pcontig) {
    Py_DECREF(pcontig);
}
"""
        return code % locals()

if has_gpua:
    @register_opt('fast_compile')
    @op_lifter([PermutohedralHashTable])
    @register_opt2([PermutohedralHashTable], 'fast_compile')
    def local_permutohedral_hash_to_gpu(op, context_name, inputs, outputs):
        return GpuHashTable()(*inputs)

def hashtable(ref, dim):
    """
    Builds the hashtable for a permutohedral lattice with d='dim' based on the
    reference image 'ref'. 'dim' must be a known scalar constant.

    Parameters
    ----------

    ref : array_like, shape (dim, H, W)
        A set of reference points to insert into the hash table.
    dim : int
        The dimensionality of the points, must be a known scalar constant.
    """

    return PermutohedralHashTable()(ref, dim)
