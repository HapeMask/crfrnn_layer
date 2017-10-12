import torch as th

#from .._ext.permutohedral import build_hash_cpu
from .._ext.permutohedral import build_hash_cuda

def make_buffers(dim, h, w, cap, cuda=False):
    buffers = [th.IntTensor(cap),           # hash_entries
               th.ShortTensor(cap, dim),    # hash_keys
               th.IntTensor(dim+1, h, w),   # neib_ents
               th.FloatTensor(dim+1, h, w), # barycentric
               th.IntTensor(cap),           # valid_entries
               th.IntTensor(1)]             # n_valid_entries
    if cuda:
        buffers = [b.cuda() for b in buffers]
    return buffers

def get_hash_cap(N, dim):
    return N*(dim+1)

class Hashtable(th.autograd.Function):
    @staticmethod
    def forward(ctx, points, _buffers=None):
        if not points.is_contiguous():
            points = points.contiguous()

        dim, h, w = points.shape
        N = h*w
        cap = get_hash_cap(N, dim)

        if _buffers is None:
            buffers = make_buffers(dim, h, w, cap, points.cuda)
        else:
            buffers = list(_buffers)

        buffers[0][:] = -1
        buffers[-1][0] = 0

        # This isn't needed in python>=3.5
        args = [points] + buffers + [cap]

        if points.is_cuda:
            build_hash_cuda(*args)
        else:
            raise NotImplementedError("Hash table currently requires CUDA support.")
            #build_hash_cpu(*args)

        return buffers

    @staticmethod
    def backward(ctx, *args, **kwargs):
        raise NotImplementedError("Cannot take the gradient of hashtable construction.")

hashtable = Hashtable.apply
