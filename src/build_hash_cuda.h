#pragma once

void build_hash_cuda(const torch::Tensor& th_points,
        torch::Tensor th_hash_entries,
        torch::Tensor th_hash_keys,
        torch::Tensor th_neib_ents,
        torch::Tensor th_barycentric,
        torch::Tensor th_valid_entries,
        torch::Tensor th_n_valid);
