import os

import numpy as np
import theano
from pygpu.gpuarray import GpuArray, SIZE
from theano.tensor.basic import as_tensor_variable, get_scalar_constant_value
from theano.scalar.basic import constant
from theano.gpuarray.type import GpuArrayType, get_context
from theano.gpuarray.basic_ops import GpuKernelBase, gpu_contiguous, Kernel
from theano.gof.op import Op
from theano.gof.graph import Apply
from theano.gof.null_type import NullType

MIN_QUAD_PROBES = 10000


class GpuHashTable(GpuKernelBase, Op):
    """Represents the construction of the hash table / slicing table for a
    permutohedral lattice built from a reference image.

    Outputs
    -----
    entries : gpuarray, shape = [table_capacity]
        An array of indices mapping a given hash value to the corresponding
        location in one of the auxiliary arrays (keys, valid entries, etc...)

    keys : gpuarray, shape = [table_capacity, d]
        An array of keys for inserted items. Since the table corresponds to a
        permutohedral lattice, the keys are points on the d+1 dimensional
        hyperplane
        :math:`\mathbf{x}\cdot\mathbf{1} = 0,\mathbf{x} \in \mathbb{Z}^{d+1}`,
        and the last dimension (or any other single dimension) is redundant.

    neighbor_entries : gpuarray, shape = [d+1, H, W]
        An array of d+1 table indices for each each pixel mapping it to its d+1
        neighboring points on the lattice.

    barycentric_coords : gpuarray, shape = [d+1, H, W]
        Barycentric coordinates within the surrounding simplex for each pixel.

    valid_entries : gpuarray, shape = [table_capacity]
        An array of entries remaining after duplicates have been pruned (the
        hash table construction may insert the same key twice in different
        locations, only the first is used).

    n_valid : array, shape = [1]
        Number of non-duplicate entries in the table (the number of non-'-1'
        values in valid_entries).
    """

    __props__ = ("context_name",)

    def __init__(self, context_name=None):
        self.context_name = context_name

    def get_params(self, node):
        return get_context(self.context_name)

    def make_node(self, points, dim):
        assert(points.ndim == 3)
        points = gpu_contiguous(as_tensor_variable(points.astype("float32")))

        dim = get_scalar_constant_value(dim)
        if "int" not in str(dim.dtype):
            raise ValueError("dim must be an integer.")

        if dim > 31:
            raise ValueError("GpuHashtable does not currently support \
dimensionality > 31.")

        dim = constant(dim, dtype="int32", name="dim")

        entries_type = GpuArrayType("int32",
                                    broadcastable=(False,),
                                    context_name=self.context_name,
                                    name="entries")
        keys_type = GpuArrayType("int16",
                                 broadcastable=(False, False),
                                 context_name=self.context_name,
                                 name="keys")
        neib_ent_type = GpuArrayType("int32",
                                     broadcastable=(False, False, False),
                                     context_name=self.context_name,
                                     name="neighbor_entries")
        bary_type = GpuArrayType("float32",
                                 broadcastable=points.type.broadcastable,
                                 context_name=self.context_name,
                                 name="barycentric_coords")

        valid_entries_type = GpuArrayType("int32",
                                          broadcastable=(False,),
                                          context_name=self.context_name,
                                          name="valid_entries")
        n_valid_type = GpuArrayType("int32",
                                    broadcastable=(False,),
                                    context_name=self.context_name,
                                    name="n_valid")

        out_vars = [entries_type(name="hash_entries"),
                    keys_type(name="hash_keys"),
                    neib_ent_type(name="neighbor_entries"),
                    bary_type(name="barycentric_coords"),
                    valid_entries_type(name="valid_entries"),
                    n_valid_type(name="n_valid")]

        # TODO: I suppose GpuHashTable should be a type like GpuHashType, and
        # the Op should return one of those instead.

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
                  "MIN_QUAD_PROBES":    str(MIN_QUAD_PROBES)}

        for k, v in consts.items():
            define_macros.append(define_template % (k, v))
            undef_macros.append(undef_template % k)

        return ''.join(define_macros), ''.join(undef_macros)

    def gpu_kernels(self, node, name):
        dim = get_scalar_constant_value(node.inputs[1])
        flags = Kernel.get_flags(node.inputs[0].dtype)

        def_macros, undef_macros = self._macros(node, name)
        hsup = (self._hash_support_code() + "\n" + self._lookup_code())

        knames = ["build_hash", "dedup", "find_valid"]
        kcodes = ["".join(open("%s%s%s.cu" %
                  (os.path.dirname(__file__), os.path.sep, kn)).readlines())
                  for kn in knames]
        kcodes = ["\n".join([def_macros, hsup, code, undef_macros])
                  for code in kcodes]

        kparams = ([GpuArray, SIZE,
                    GpuArray, SIZE,
                    GpuArray, SIZE,
                    GpuArray, SIZE, GpuArray, SIZE,
                    SIZE, SIZE],

                   [GpuArray, SIZE, GpuArray, SIZE, SIZE],

                   [GpuArray, SIZE, GpuArray, SIZE, GpuArray, SIZE, SIZE])

        return [Kernel(code=kcode, name="%s_%d" % (kname, dim),
                params=kparams, flags=flags)
                for kcode, kname, kparams in zip(kcodes, knames, kparams)]

    @staticmethod
    def _hash_support_code():
        code = """
__device__ unsigned int DIM_SPECIFIC(hash)(const short* key) {
    unsigned int h = 0;
    for (int i=0; i < KEY_DIM; ++i) {
        h ^= ((unsigned int)key[i]) << ((31/KEY_DIM)*i);
    }
    return h;
}

__device__ bool DIM_SPECIFIC(key_cmp)(const short* key1, const short* key2) {
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
                    os.path.sep+"lookup.cu").readlines())

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
        points = inputs[0]
        entries, keys, neib_ents, barycentric, valid_entries, n_valid = outputs

        dim = get_scalar_constant_value(node.inputs[1])

        fail = sub["fail"]
        ctx = sub["params"]
        sync = bool(theano.config.gpuarray.sync)
        kname_build = "k_build_hash_%d" % dim
        kname_dedup = "k_dedup_%d" % dim
        kname_fve = "k_find_valid_%d" % dim

        code = """
int err = GA_NO_ERROR;

size_t point_dims[3];
size_t entries_dim[1];
size_t keys_dims[2];
size_t neib_ents_dims[3];
size_t barycentric_dims[3];
size_t valid_entries_dim[1];

size_t n_valid_dim[1];
n_valid_dim[0] = 1;

point_dims[0] = PyGpuArray_DIMS(%(points)s)[0];
point_dims[1] = PyGpuArray_DIMS(%(points)s)[1];
point_dims[2] = PyGpuArray_DIMS(%(points)s)[2];

size_t N = point_dims[1] * point_dims[2];
size_t cap = N*(point_dims[0]+1);

size_t ls_N, gs_N, ls_cap, gs_cap;

if(point_dims[0] != %(dim)s) {
    PyErr_Format(PyExc_ValueError,
        "hashtable error: incorrect input dim 0.\\nExpected %(dim)s got %%d",
        point_dims[0]);
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

if(!valid_output_ptr(%(entries)s, GA_INT, 1, entries_dim)) {
    Py_XDECREF(%(entries)s);
    %(entries)s = pygpu_empty(1, entries_dim, GA_INT, GA_C_ORDER, %(ctx)s,
        Py_None);
}
if(!valid_output_ptr(%(keys)s, GA_SHORT, 2, keys_dims)) {
    Py_XDECREF(%(keys)s);
    %(keys)s = pygpu_zeros(2, keys_dims, GA_SHORT, GA_C_ORDER, %(ctx)s,
        Py_None);
}
if(!valid_output_ptr(%(neib_ents)s, GA_INT, 3, neib_ents_dims)) {
    Py_XDECREF(%(neib_ents)s);
    %(neib_ents)s = pygpu_zeros(3, neib_ents_dims, GA_INT, GA_C_ORDER, %(ctx)s,
        Py_None);
}
if(!valid_output_ptr(%(barycentric)s, GA_FLOAT, 3, barycentric_dims)) {
    Py_XDECREF(%(barycentric)s);
    %(barycentric)s = pygpu_zeros(3, barycentric_dims, GA_FLOAT, GA_C_ORDER,
        %(ctx)s, Py_None);
}
if(!valid_output_ptr(%(valid_entries)s, GA_INT, 1, valid_entries_dim)) {
    Py_XDECREF(%(valid_entries)s);
    %(valid_entries)s = pygpu_zeros(1, valid_entries_dim, GA_INT, GA_C_ORDER,
        %(ctx)s, Py_None);
}
if(!valid_output_ptr(%(n_valid)s, GA_INT, 1, n_valid_dim)) {
    Py_XDECREF(%(n_valid)s);
    %(n_valid)s = pygpu_zeros(1, n_valid_dim, GA_INT, GA_C_ORDER,
        %(ctx)s, Py_None);
}

if (!(%(entries)s && %(keys)s && %(neib_ents)s && %(barycentric)s &&
    %(valid_entries)s)) {

    PyErr_Format(PyExc_MemoryError,
            "error building hash table: failed to allocate output storage.");
    %(fail)s;
}

GpuArray_memset(&%(entries)s->ga, -1);
GpuArray_memset(&%(n_valid)s->ga, 0);

gs_N = ls_N = 0;
GpuKernel_sched(&%(kname_build)s, N, &gs_N, &ls_N);
gs_N = N / ls_N;
if (ls_N*gs_N < N) { ++gs_N; }

err = build_hash_%(dim)s_call(1, &gs_N, &ls_N, 0,
    %(points)s->ga.data, %(points)s->ga.offset / sizeof(float),
    %(entries)s->ga.data, %(entries)s->ga.offset / sizeof(int),
    %(keys)s->ga.data, %(keys)s->ga.offset / sizeof(short),
    %(neib_ents)s->ga.data, %(neib_ents)s->ga.offset / sizeof(int),
    %(barycentric)s->ga.data, %(barycentric)s->ga.offset / sizeof(float),
    cap, N);

if(err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
        "gpuarray error building hash table:\\n%%s.\\n",
        GpuKernel_error(&%(kname_build)s, err));
    %(fail)s;
}

GpuArray_sync(&%(entries)s->ga);
GpuArray_sync(&%(keys)s->ga);

gs_cap = ls_cap = 0;
GpuKernel_sched(&%(kname_dedup)s, cap, &gs_cap, &ls_cap);
gs_cap = cap / ls_cap;
if (ls_cap*gs_cap < cap) { ++gs_cap; }

err = dedup_%(dim)s_call(1, &gs_cap, &ls_cap, 0,
    %(entries)s->ga.data, %(entries)s->ga.offset / sizeof(int),
    %(keys)s->ga.data, %(keys)s->ga.offset / sizeof(short),
    cap);

if(err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
        "gpuarray error cleaning hash table:\\n%%s.\\n",
        GpuKernel_error(&%(kname_dedup)s, err));
    %(fail)s;
}

GpuArray_sync(&%(entries)s->ga);
GpuArray_sync(&%(keys)s->ga);

err = find_valid_%(dim)s_call(1, &gs_cap, &ls_cap, 0,
    %(entries)s->ga.data, %(entries)s->ga.offset / sizeof(int),
    %(valid_entries)s->ga.data, %(valid_entries)s->ga.offset / sizeof(int),
    %(n_valid)s->ga.data, %(n_valid)s->ga.offset / sizeof(int),
    cap);

if(err != GA_NO_ERROR) {
    PyErr_Format(PyExc_RuntimeError,
        "gpuarray error counting valid hash entries:\\n%%s.\\n",
        GpuKernel_error(&%(kname_fve)s, err));
    %(fail)s;
}

GpuArray_sync(&%(entries)s->ga);
GpuArray_sync(&%(keys)s->ga);
GpuArray_sync(&%(neib_ents)s->ga);
GpuArray_sync(&%(barycentric)s->ga);
GpuArray_sync(&%(n_valid)s->ga);
"""
        return code % locals()


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

    return GpuHashTable()(ref, dim)
