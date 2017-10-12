#include <THC/THC.h>
#include <cuda_runtime.h>

#include "gfilt_kernel.h"

extern THCState* state;

int gfilt_cuda(const THCudaTensor* th_values, THCudaTensor* th_output,
        THCudaTensor* th_tmp_vals_1, THCudaTensor* th_tmp_vals_2,
        const THCudaIntTensor* th_hash_entries, const THCudaShortTensor* th_hash_keys,
        const THCudaIntTensor* th_neib_ents, const THCudaTensor* th_barycentric,
        const THCudaIntTensor* th_valid_entries, int n_valid,
        size_t hash_cap, size_t ref_dim) {

    const float* values = THCudaTensor_data(state, th_values);
    float* output = THCudaTensor_data(state, th_output);
    float* tmp_vals_1 = THCudaTensor_data(state, th_tmp_vals_1);
    float* tmp_vals_2 = THCudaTensor_data(state, th_tmp_vals_2);
    const int* hash_entries = THCudaIntTensor_data(state, th_hash_entries);
    const short* hash_keys = THCudaShortTensor_data(state, th_hash_keys);
    const int* neib_ents = THCudaIntTensor_data(state, th_neib_ents);
    const float* barycentric = THCudaTensor_data(state, th_barycentric);
    const int* valid_entries = THCudaIntTensor_data(state, th_valid_entries);

    const size_t val_dim = THCudaTensor_size(state, th_values, 0);
    const size_t N = THCudaTensor_size(state, th_values, 1) * THCudaTensor_size(state, th_values, 2);

    cudaStream_t stream = THCState_getCurrentStream(state);
    cudaError_t err;

    call_gfilt_kernels(values, output, tmp_vals_1, tmp_vals_2, hash_entries,
                       hash_keys, neib_ents, barycentric, valid_entries,
                       n_valid, hash_cap, N, ref_dim, val_dim, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "build_hash CUDA kernel failure: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return 1;
}

