#include <torch/extension.h>
#include <cuda_runtime.h>

#include "build_hash_kernel.h"

int build_hash_cuda(const torch::Tensor& th_points,
        torch::Tensor th_hash_entries,
        torch::Tensor th_hash_keys,
        torch::Tensor th_neib_ents,
        torch::Tensor th_barycentric,
        torch::Tensor th_valid_entries,
        torch::Tensor th_n_valid,
        size_t hash_cap) {

    const float* points = th_points.data_ptr<float>();
    int* hash_entries = th_hash_entries.data_ptr<int>();
    short* hash_keys = th_hash_keys.data_ptr<short>();
    int* neib_ents = th_neib_ents.data_ptr<int>();
    float* barycentric = th_barycentric.data_ptr<float>();
    int* valid_entries = th_valid_entries.data_ptr<int>();
    int* n_valid = th_n_valid.data_ptr<int>();

    const size_t dim = th_points.sizes()[0];
    const size_t N = th_points.sizes()[1] * th_points.sizes()[2];

    cudaError_t err;
    call_build_hash_kernels(points, hash_entries, hash_keys, neib_ents, barycentric, valid_entries, n_valid, hash_cap, N, dim, 0);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "build_hash CUDA kernel failure: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    return 1;
}
