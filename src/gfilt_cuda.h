#pragma once

void gfilt_cuda(const torch::Tensor& th_values, torch::Tensor th_output,
        torch::Tensor th_tmp_vals_1, torch::Tensor th_tmp_vals_2,
        const torch::Tensor& th_hash_entries, const torch::Tensor& th_hash_keys,
        const torch::Tensor& th_neib_ents, const torch::Tensor& th_barycentric,
        const torch::Tensor& th_valid_entries, const torch::Tensor& n_valid,
        int64_t ref_dim, bool reverse);
