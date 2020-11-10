#include <stdio.h>

#include "hash_fns.cuh"
#include "gfilt_kernel.h"

template <size_t ref_dim, size_t val_dim>
__global__ void splat(const float* values,
                      const float* barycentric,
                      const int* hash_entries,
                      const int* neib_ents,
                      float* hash_values, size_t N) {

    const size_t idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx >= N) { return; }

    float local_v[val_dim];
    float local_b[ref_dim+1];

    for(int i=0; i<val_dim; ++i) { local_v[i] = values[idx + N*i]; }
    for(int i=0; i<(ref_dim+1); ++i) { local_b[i] = barycentric[idx + N*i]; }

    // Splat this point onto each vertex of its surrounding simplex.
    for(int k=0; k<ref_dim+1; ++k) {
        const int& ind = hash_entries[neib_ents[idx + N*k]];
        float* hv = &hash_values[ind*val_dim];

        for(int i=0; i<val_dim; ++i) {
            atomicAdd(&hv[i], local_b[k] * local_v[i]);
        }
    }
}

template <size_t ref_dim, size_t val_dim>
__global__ void blur(float* out,
                     const int* hash_entries,
                     const int* valid_entries,
                     const short* hash_keys,
                     const float* hash_values,
                     size_t hash_cap, int n_valid, size_t axis) {

    const size_t idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx >= n_valid) { return; }

    float local_out[val_dim];

    // The local key storage needs the normally-ignored value at the end so
    // that key[axis] is always a valid memory access.
    //short key[ref_dim];
    short key[ref_dim+1];

    const int& ind_c = valid_entries[idx];
    for(int i=0; i<val_dim; ++i) { local_out[i] = 0; }

    const short* key_c = &hash_keys[ind_c*ref_dim];

    for(int i=0; i<ref_dim; ++i) { key[i] = key_c[i] + 1; }
    if(axis < ref_dim) { key[axis] = key_c[axis] - ref_dim; }
    key[axis] = key_c[axis] - ref_dim;
    const int ind_l = hash_lookup<ref_dim>(hash_entries, hash_keys, hash_cap, key);

    for(int i=0; i<ref_dim; ++i) { key[i] = key_c[i] - 1; }
    if(axis < ref_dim) { key[axis] = key_c[axis] + ref_dim; }
    key[axis] = key_c[axis] + ref_dim;
    const int ind_r = hash_lookup<ref_dim>(hash_entries, hash_keys, hash_cap, key);

    if(ind_l >= 0 && ind_r >= 0) {
        for(int i=0; i<val_dim; ++i) {
            local_out[i] = (hash_values[ind_l*val_dim + i] +
                     2*hash_values[ind_c*val_dim + i] +
                     hash_values[ind_r*val_dim + i]) * 0.25f;
        }
    } else if(ind_l >= 0) {
        for(int i=0; i<val_dim; ++i) {
            local_out[i] = (hash_values[ind_l*val_dim + i] +
                     2*hash_values[ind_c*val_dim + i]) * 0.25f;
        }
    } else if(ind_r >= 0) {
        for(int i=0; i<val_dim; ++i) {
            local_out[i] = (hash_values[ind_r*val_dim + i] +
                     2*hash_values[ind_c*val_dim + i]) * 0.25f;
        }
    } else {
        for(int i=0; i<val_dim; ++i) {
            local_out[i] = hash_values[ind_c*val_dim + i] * 0.5f;
        }
    }

    for(int i=0; i<val_dim; ++i) { out[ind_c*val_dim + i] = local_out[i]; }
}

template <size_t ref_dim, size_t val_dim>
__global__ void slice(float* out,
                      const float* barycentric,
                      const int* hash_entries,
                      const int* neib_ents,
                      const float* hash_values, int N) {

    const size_t idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx >= N) { return; }

    float local_out[val_dim];

    for(int i=0; i<val_dim; ++i) { local_out[i] = 0; }

    // Gather values from each of the surrounding simplex vertices.
    for(int k=0; k<ref_dim+1; ++k) {
        const int& ind = hash_entries[neib_ents[idx + N*k]];
        const float* hv = &hash_values[ind*val_dim];

        for(int i=0; i<val_dim; ++i) {
            local_out[i] += barycentric[idx + N*k] * hv[i];
        }
    }

    for(int i=0; i<val_dim; ++i) { out[idx + N*i] = local_out[i]; }
}

template<size_t ref_dim, size_t val_dim>
void _call_gfilt_kernels(const float* values, float* output,
                         float* tmp_vals_1, float* tmp_vals_2,
                         const int* hash_entries, const short* hash_keys,
                         const int* neib_ents, const float* barycentric,
                         const int* valid_entries, int n_valid,
                         size_t hash_cap, size_t N, bool reverse, cudaStream_t stream) {

    splat<ref_dim, val_dim><<<cuda_gridsize(N), BLOCK, 0, stream>>>(values,
            barycentric, hash_entries, neib_ents, tmp_vals_1, N);

    float* tmp_swap;

    for(int ax=reverse ? ref_dim : 0;
        ax <= ref_dim && ax >= 0;
        reverse ? --ax : ++ax
    ) {
        blur<ref_dim, val_dim><<<cuda_gridsize(n_valid), BLOCK, 0,
            stream>>>(tmp_vals_2, hash_entries, valid_entries, hash_keys,
                      tmp_vals_1, hash_cap, n_valid, ax);

        tmp_swap = tmp_vals_1;
        tmp_vals_1 = tmp_vals_2;
        tmp_vals_2 = tmp_swap;
    }

    slice<ref_dim, val_dim><<<cuda_gridsize(N), BLOCK, 0, stream>>>(output, barycentric, hash_entries, neib_ents, tmp_vals_2, N);
}

void call_gfilt_kernels(const float* values, float* output,
                        float* tmp_vals_1, float* tmp_vals_2,
                        const int* hash_entries, const short* hash_keys,
                        const int* neib_ents, const float* barycentric,
                        const int* valid_entries, int n_valid,
                        size_t hash_cap, size_t N, size_t ref_dim,
                        size_t val_dim, bool reverse, cudaStream_t stream) {

#include "gfilt_dispatch_table.h"
}
