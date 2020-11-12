import torch as th

from permutohedral_ext import build_hash_cuda

def make_hash_buffers(dim, h, w, cap, dev):
    return [-th.ones(cap, dtype=th.int32, device=dev),         # hash_entries
            th.zeros(cap, dim, dtype=th.int16, device=dev),    # hash_keys
            th.zeros(dim+1, h, w, dtype=th.int32, device=dev), # neib_ents
            th.zeros(dim+1, h, w, device=dev),                 # barycentric
            th.zeros(cap, dtype=th.int32, device=dev),         # valid_entries
            th.tensor(1).int().to(device=dev)]                 # n_valid_entries

def get_hash_cap(N, dim):
    return N*(dim+1)

def make_hashtable(points):
    if not points.is_contiguous():
        points = points.contiguous()

    dim, h, w = points.shape
    N = h*w
    cap = get_hash_cap(N, dim)

    buffers = make_hash_buffers(dim, h, w, cap, points.device)
    if points.is_cuda:
        build_hash_cuda(points, *buffers, cap)
    else:
        raise NotImplementedError("Hash table currently requires CUDA support.")
        #build_hash_cpu(points, *buffers, cap)

    return buffers
