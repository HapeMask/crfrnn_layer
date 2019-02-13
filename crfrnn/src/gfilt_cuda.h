#pragma once

int gfilt_cuda(const at::Tensor& th_values, at::Tensor th_output,
        at::Tensor th_tmp_vals_1, at::Tensor th_tmp_vals_2,
        const at::Tensor& th_hash_entries, const at::Tensor& th_hash_keys,
        const at::Tensor& th_neib_ents, const at::Tensor& th_barycentric,
        const at::Tensor& th_valid_entries, int n_valid,
        size_t hash_cap, size_t ref_dim);
