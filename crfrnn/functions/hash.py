import torch as th

from permutohedral import build_hash_cuda

def make_hash_buffers(dim, h, w, cap, dev):
    return [th.zeros(cap, dtype=th.int32, device=dev),         # hash_entries
            th.zeros(cap, dim, dtype=th.int16, device=dev),    # hash_keys
            th.zeros(dim+1, h, w, dtype=th.int32, device=dev), # neib_ents
            th.zeros(dim+1, h, w, device=dev),                 # barycentric
            th.zeros(cap, dtype=th.int32, device=dev),         # valid_entries
            th.tensor(1).int().to(device=dev)]                 # n_valid_entries

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
            buffers = make_hash_buffers(dim, h, w, cap, points.device)
        else:
            buffers = list(_buffers)

        buffers[0].fill_(-1)
        buffers[-1].fill_(0)

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
