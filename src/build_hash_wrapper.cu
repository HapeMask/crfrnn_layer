#include <torch/extension.h>
#include <cuda_runtime.h>

#include "build_hash_kernel.h"
#include "common.h"

void build_hash_cuda(const torch::Tensor& th_points,
        torch::Tensor th_hash_entries,
        torch::Tensor th_hash_keys,
        torch::Tensor th_neib_ents,
        torch::Tensor th_barycentric,
        torch::Tensor th_valid_entries,
        torch::Tensor th_n_valid) {

    CHECK_INPUT(th_points)
    CHECK_INPUT(th_hash_entries)
    CHECK_INPUT(th_hash_keys)
    CHECK_INPUT(th_neib_ents)
    CHECK_INPUT(th_barycentric)
    CHECK_INPUT(th_valid_entries)
    CHECK_CONTIGUOUS(th_n_valid)

    CHECK_4DIMS(th_points)
    CHECK_2DIMS(th_hash_entries)
    CHECK_3DIMS(th_hash_keys)
    CHECK_4DIMS(th_neib_ents)
    CHECK_4DIMS(th_barycentric)
    CHECK_2DIMS(th_valid_entries)
    CHECK_2DIMS(th_n_valid)

    const float* points = DATA_PTR(th_points, float);
    int* hash_entries = DATA_PTR(th_hash_entries, int);
    short* hash_keys = DATA_PTR(th_hash_keys, short);
    int* neib_ents = DATA_PTR(th_neib_ents, int);
    float* barycentric = DATA_PTR(th_barycentric, float);
    int* valid_entries = DATA_PTR(th_valid_entries, int);
    int* n_valid = DATA_PTR(th_n_valid, int);

    const int B = th_points.size(0);
    const int dim = th_points.size(1);
    const int H = th_points.size(2);
    const int W = th_points.size(3);
    const int hash_cap = th_hash_entries.size(1);

    cudaError_t err;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    for (int b=0; b < B; ++b) {
        call_build_hash_kernels(
            points + (b * dim * H * W),
            hash_entries + (b * hash_cap),
            hash_keys + (b * hash_cap * dim),
            neib_ents + (b * (dim + 1) * H * W),
            barycentric + (b * (dim + 1) * H * W),
            valid_entries + (b * hash_cap),
            n_valid + b,
            hash_cap, H * W, dim, stream
        );

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "build_hash CUDA kernel failure: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}
