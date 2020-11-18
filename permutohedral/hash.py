import torch as th

import permutohedral_ext

th.ops.load_library(permutohedral_ext.__file__)
build_hash_cuda = th.ops.permutohedral_ext.build_hash_cuda


def make_hash_buffers(b, dim, h, w, cap, dev):
    return [
        -th.ones(b, cap, dtype=th.int32, device=dev),  # hash_entries
        th.zeros(b, cap, dim, dtype=th.int16, device=dev),  # hash_keys
        th.zeros(b, dim + 1, h, w, dtype=th.int32, device=dev),  # neib_ents
        th.zeros(b, dim + 1, h, w, device=dev),  # barycentric
        th.zeros(b, cap, dtype=th.int32, device=dev),  # valid_entries
        th.zeros(b, 1).int().to(device=dev),  # n_valid_entries
    ]


def get_hash_cap(N, dim):
    return N * (dim + 1)


def make_hashtable(points):
    b, dim, h, w = points.shape
    N = h * w
    cap = get_hash_cap(N, dim)

    buffers = make_hash_buffers(b, dim, h, w, cap, points.device)
    if points.is_cuda:
        build_hash_cuda(points.contiguous(), *buffers)
    else:
        raise NotImplementedError("Hash table currently requires CUDA support.")
        # build_hash_cpu(points.contiguous(), *buffers, cap)

    return buffers
