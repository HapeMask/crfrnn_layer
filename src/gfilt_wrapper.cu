#include <torch/extension.h>
#include <cuda_runtime.h>

#include "gfilt_kernel.h"
#include "common.h"

void gfilt_cuda(const torch::Tensor& th_values, torch::Tensor th_output,
        torch::Tensor th_tmp_vals_1, torch::Tensor th_tmp_vals_2,
        const torch::Tensor& th_hash_entries, const torch::Tensor& th_hash_keys,
        const torch::Tensor& th_neib_ents, const torch::Tensor& th_barycentric,
        const torch::Tensor& th_valid_entries, const torch::Tensor& th_n_valid,
        int64_t ref_dim, bool reverse) {

    CHECK_4DIMS(th_values)
    CHECK_4DIMS(th_output)
    CHECK_2DIMS(th_tmp_vals_1)
    CHECK_2DIMS(th_tmp_vals_2)

    CHECK_2DIMS(th_hash_entries)
    CHECK_3DIMS(th_hash_keys)
    CHECK_4DIMS(th_neib_ents)
    CHECK_4DIMS(th_barycentric)
    CHECK_2DIMS(th_valid_entries)
    CHECK_CONTIGUOUS(th_n_valid)

    const float* values = DATA_PTR(th_values, float);
    float* output = DATA_PTR(th_output, float);
    float* tmp_vals_1 = DATA_PTR(th_tmp_vals_1, float);
    float* tmp_vals_2 = DATA_PTR(th_tmp_vals_2, float);
    const int* hash_entries = DATA_PTR(th_hash_entries, int);
    const short* hash_keys = DATA_PTR(th_hash_keys, short);
    const int* neib_ents = DATA_PTR(th_neib_ents, int);
    const float* barycentric = DATA_PTR(th_barycentric, float);
    const int* valid_entries = DATA_PTR(th_valid_entries, int);
    int* n_valid = DATA_PTR(th_n_valid, int);

    const int B = th_values.size(0);
    const int H = th_values.size(2);
    const int W = th_values.size(3);
    const int val_dim = th_values.size(1);
    const int hash_cap = th_hash_entries.size(1);

    cudaError_t err;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    printf("B: %d\n", B);
    printf("H: %d\n", H);
    printf("W: %d\n", W);
    printf("rd: %ld\n", ref_dim);
    printf("vd: %d\n", val_dim);
    printf("n valid: %d\n", n_valid[0]);


    for (int b=0; b < B; ++b) {
        th_tmp_vals_1.fill_(0.f);
        th_tmp_vals_2.fill_(0.f);

        call_gfilt_kernels(values,
                output + (b * (val_dim + 1) * H  * W),
                tmp_vals_1,
                tmp_vals_2,
                hash_entries + (b * hash_cap),
                hash_keys + (b * hash_cap * ref_dim),
                neib_ents + (b * (ref_dim + 1) * H * W),
                barycentric + (b * (ref_dim + 1) * H * W),
                valid_entries + (b * hash_cap),
                n_valid[b],
                hash_cap,
                H * W,
                ref_dim,
                val_dim,
                reverse,
                stream);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "build_hash CUDA kernel failure: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}
