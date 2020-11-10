#include <torch/extension.h>
#include <cuda_runtime.h>

#include "gfilt_kernel.h"

int gfilt_cuda(const torch::Tensor& th_values, torch::Tensor th_output,
        torch::Tensor th_tmp_vals_1, torch::Tensor th_tmp_vals_2,
        const torch::Tensor& th_hash_entries, const torch::Tensor& th_hash_keys,
        const torch::Tensor& th_neib_ents, const torch::Tensor& th_barycentric,
        const torch::Tensor& th_valid_entries, int n_valid,
        size_t hash_cap, size_t ref_dim, bool reverse) {

    const float* values = th_values.data_ptr<float>();
    float* output = th_output.data_ptr<float>();
    float* tmp_vals_1 = th_tmp_vals_1.data_ptr<float>();
    float* tmp_vals_2 = th_tmp_vals_2.data_ptr<float>();
    const int* hash_entries = th_hash_entries.data_ptr<int>();
    const short* hash_keys = th_hash_keys.data_ptr<short>();
    const int* neib_ents = th_neib_ents.data_ptr<int>();
    const float* barycentric = th_barycentric.data_ptr<float>();
    const int* valid_entries = th_valid_entries.data_ptr<int>();

    const size_t val_dim = th_values.sizes()[0];
    const size_t N = th_values.sizes()[1] * th_values.sizes()[2];

    cudaError_t err;

    call_gfilt_kernels(values, output, tmp_vals_1, tmp_vals_2, hash_entries,
                       hash_keys, neib_ents, barycentric, valid_entries,
                       n_valid, hash_cap, N, ref_dim, val_dim, reverse, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "build_hash CUDA kernel failure: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return 1;
}
