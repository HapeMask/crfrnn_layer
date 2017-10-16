#include <THC/THC.h>
#include <cuda_runtime.h>

#include "build_hash_kernel.h"

extern THCState* state;

int build_hash_cuda(const THCudaTensor* th_points,
        THCudaIntTensor* th_hash_entries,
        THCudaShortTensor* th_hash_keys,
        THCudaIntTensor* th_neib_ents,
        THCudaTensor* th_barycentric,
        THCudaIntTensor* th_valid_entries,
        THCudaIntTensor* th_n_valid,
        size_t hash_cap) {

    const float* points = THCudaTensor_data(state, th_points);
    int* hash_entries = THCudaIntTensor_data(state, th_hash_entries);
    short* hash_keys = THCudaShortTensor_data(state, th_hash_keys);
    int* neib_ents = THCudaIntTensor_data(state, th_neib_ents);
    float* barycentric = THCudaTensor_data(state, th_barycentric);
    int* valid_entries = THCudaIntTensor_data(state, th_valid_entries);
    int* n_valid = THCudaIntTensor_data(state, th_n_valid);

    const size_t dim = THCudaTensor_size(state, th_points, 0);
    const size_t N = THCudaTensor_size(state, th_points, 1) * THCudaTensor_size(state, th_points, 2);

    cudaStream_t stream = THCState_getCurrentStream(state);
    cudaError_t err;

    call_build_hash_kernels(points, hash_entries, hash_keys, neib_ents, barycentric, valid_entries, n_valid, hash_cap, N, dim, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "build_hash CUDA kernel failure: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return 1;
}
