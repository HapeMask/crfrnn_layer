#pragma once

int build_hash_cuda(const at::Tensor& th_points,
        at::Tensor th_hash_entries,
        at::Tensor th_hash_keys,
        at::Tensor th_neib_ents,
        at::Tensor th_barycentric,
        at::Tensor th_valid_entries,
        at::Tensor th_n_valid,
        size_t hash_cap);
