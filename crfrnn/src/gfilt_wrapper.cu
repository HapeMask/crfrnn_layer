#include <torch/extension.h>
#include <cuda_runtime.h>

#include "gfilt_kernel.h"

int gfilt_cuda(const at::Tensor& th_values, at::Tensor th_output,
        at::Tensor th_tmp_vals_1, at::Tensor th_tmp_vals_2,
        const at::Tensor& th_hash_entries, const at::Tensor& th_hash_keys,
        const at::Tensor& th_neib_ents, const at::Tensor& th_barycentric,
        const at::Tensor& th_valid_entries, int n_valid,
        size_t hash_cap, size_t ref_dim) {

    const float* values = th_values.data<float>();
    float* output = th_output.data<float>();
    float* tmp_vals_1 = th_tmp_vals_1.data<float>();
    float* tmp_vals_2 = th_tmp_vals_2.data<float>();
    const int* hash_entries = th_hash_entries.data<int>();
    const short* hash_keys = th_hash_keys.data<short>();
    const int* neib_ents = th_neib_ents.data<int>();
    const float* barycentric = th_barycentric.data<float>();
    const int* valid_entries = th_valid_entries.data<int>();

    const size_t val_dim = th_values.sizes()[0];
    const size_t N = th_values.sizes()[1] * th_values.sizes()[2];

    cudaError_t err;

    call_gfilt_kernels(values, output, tmp_vals_1, tmp_vals_2, hash_entries,
                       hash_keys, neib_ents, barycentric, valid_entries,
                       n_valid, hash_cap, N, ref_dim, val_dim, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "build_hash CUDA kernel failure: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return 1;
}
