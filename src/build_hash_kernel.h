#ifndef _BHK_H
#define _BHK_H

#define BLOCK 256

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
