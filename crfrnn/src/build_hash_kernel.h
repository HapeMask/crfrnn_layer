#ifndef _BHK_H
#define _BHK_H

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

void call_build_hash_kernels(const float* points,
                int* hash_entries,
                short* hash_keys,
                int* neib_ents,
                float* barycentric,
                int* valid_entries,
                int* n_valid,
                size_t hash_cap,
                size_t N, size_t dim,
                cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
