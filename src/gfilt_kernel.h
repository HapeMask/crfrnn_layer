#ifndef _GFK_H
#define _GFK_H

#define BLOCK 256

#ifdef __cplusplus
extern "C" {
#endif

void call_gfilt_kernels(const float* values, float* output,
                        float* tmp_vals_1, float* tmp_vals_2,
                        const int* hash_entries, const short* hash_keys,
                        const int* neib_ents, const float* barycentric,
                        const int* valid_entries, int n_valid,
                        size_t hash_cap, size_t N, size_t ref_dim,
                        size_t val_dim, bool reverse, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
