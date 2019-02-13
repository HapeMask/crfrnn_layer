#ifndef _GFK_H
#define _GFK_H

#define BLOCK 256

#define CUDA_SAFE_CALL(call) \
do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA kernel failure in file %s at line %i: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(-1); \
    } \
} while (0)

#ifdef __cplusplus
extern "C" {
#endif

void call_gfilt_kernels(const float* values, float* output,
                        float* tmp_vals_1, float* tmp_vals_2,
                        const int* hash_entries, const short* hash_keys,
                        const int* neib_ents, const float* barycentric,
                        const int* valid_entries, int n_valid,
                        size_t hash_cap, size_t N, size_t ref_dim,
                        size_t val_dim, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
