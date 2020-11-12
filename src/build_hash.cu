#include <stdio.h>

#include "hash_fns.cuh"
#include "build_hash_kernel.h"

template <size_t dim>
__device__ int hash_insert(int* entries, short* keys, size_t capacity, const short* key) {
    unsigned int h = hash<dim>(key) % capacity;
    const unsigned int init_h = h;
    const size_t max_it = min(MIN_QUAD_PROBES, capacity);
    int* entry = &entries[h];

    // First try quadratic probing for a fixed number of iterations, then fall
    // back to linear probing.
    for(int iter=0; iter<max_it; ++iter) {
        entry = &entries[h];
        const int ret = atomicCAS(entry, -1, -2);

        if (ret == -1) {
            // This thread got the lock and it was empty, fill it.
            *(entry) = h;
            for(int i=0; i<dim; ++i) { keys[h*dim + i] = key[i]; }
            return h;
        } else if (ret >= 0 && key_cmp<dim>(&keys[ret*dim], key)){
            // If another thread already inserted the same key, return the
            // entry for that key.
            return ret;
        }

        h = (init_h + iter*iter) % capacity;
    }

    for(int iter=0; iter<capacity; ++iter) {
        entry = &entries[h];
        const int ret = atomicCAS(entry, -1, -2);

        if (ret == -1) {
            // This thread got the lock and it was empty, fill it.
            *(entry) = h;
            for(int i=0; i<dim; ++i) { keys[h*dim + i] = key[i]; }
            return h;
        } else if (ret >= 0 && key_cmp<dim>(&keys[ret*dim], key)){
            // If another thread already inserted the same key, return the
            // entry for that key.
            return ret;
        }

        h = (h+1) % capacity;
    }

    // We wrapped around without finding a free slot, the table is full.
    return -1;
}

template <size_t dim>
__device__ __inline__ void embed(const float* f, float* e, size_t N) {
    e[dim] = -sqrt(dim / (dim + 1.f)) * f[N*(dim - 1)];
    for(int i=dim - 1; i > 0; --i) {
        e[i] = f[N * i] / sqrt((i + 1.f) / (i + 2.f)) + e[i + 1] - sqrt(i/(i + 1.f)) * f[N * (i - 1)];
    }
    e[0] = f[0] / sqrt(0.5f) + e[1];
}

template <size_t dim>
__device__ __inline__ short round2mult(float f) {
    const float s = f / (dim+1.f);
    const float lo = floor(s) * (dim+1.f);
    const float hi = ceil(s) * (dim+1.f);
    return ((hi-f) > (f-lo)) ? lo : hi;
}

template <size_t dim>
__device__ __inline__ void ranksort_diff(const float* v1, const short* v2, short* rank) {
    for(int i=0; i<dim+1; ++i) {
        rank[i] = 0;
        const float di = v1[i] - v2[i];
        for(int j=0; j<dim+1; ++j) {
            const float dj = v1[j] - v2[j];
            if (di < dj || (di==dj && i>j)) { ++rank[i]; }
        }
    }
}

template <size_t dim>
__device__ __inline__ short canonical_coord(int k, int i) {
    return (i < (dim+1-k)) ?  k : (k - (dim+1));
}

template <size_t dim>
__global__ void build_hash(const float* points,
                int* hash_entries,
                short* hash_keys,
                int* neib_ents,
                float* barycentric,
                size_t hash_cap,
                size_t N) {

    const size_t idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        short rounded[dim+1];
        short rank[dim+1];
        float embedded[dim+1];

        const float* p = &points[idx];
        float* b = &barycentric[idx];

        embed<dim>(p, embedded, N);

        short sum = 0;
        for(int i=0; i<dim+1; ++i) {
            const short r = round2mult<dim>(embedded[i]);
            rounded[i] = r;
            sum += r;
        }
        sum /= (short)(dim+1);

        // Compute rank(embedded - rounded), decreasing order
        ranksort_diff<dim>(embedded, rounded, rank);

        // Walk the point back onto H_d (Lemma 2.9 in permutohedral_techreport.pdf)
        if (sum > 0) {
            for(int i=0; i<dim+1; ++i) {
                if(rank[i] >= dim+1-sum) { rounded[i] -= dim+1; }
            }
        } else if (sum < 0) {
            for(int i=0; i<dim+1; ++i) {
                if(rank[i] < -sum) { rounded[i] += dim+1; }
            }
        }

        // Re-ompute rank(embedded - rounded), decreasing order. TODO: The
        // reference code doesn't do this and does some other things instead.
        // Find out why?
        ranksort_diff<dim>(embedded, rounded, rank);

        // The temporary key has 1 extra dimension. Normally we ignore the last
        // entry in the key because they sum to 0, but we can use the key as swap
        // space to invert the sorting permutation.
        short tmpkey[dim+1];
        for(int k=0; k<dim+1; ++k) {
            for(int i=0; i<dim; ++i) {
                tmpkey[i] = canonical_coord<dim>(k, rank[i]) + rounded[i];
            }

            const int ind = hash_insert<dim>(hash_entries, hash_keys, hash_cap, tmpkey);
            assert(ind >= 0);
            neib_ents[idx + N*k] = ind;
        }

        // Rank is equivalent to the inverse permutation for sorting in
        // decreasing order, invert it here to compute barycentric coordinates
        // (which requires the non-inverted permutation).
        for(int i=0; i<dim+1; ++i) { tmpkey[rank[i]] = i; }
        for(int i=0; i<dim+1; ++i) { rank[i] = tmpkey[i]; }

        b[0] = 1 - ((embedded[rank[0]] - rounded[rank[0]]) -
                    (embedded[rank[dim]] - rounded[rank[dim]])) / (dim + 1.f);
        for(int i=1; i<dim+1; ++i) {
            b[N*i] = ((embedded[rank[dim - i]] - rounded[rank[dim - i]])  -
                      (embedded[rank[dim + 1 - i]] - rounded[rank[dim + 1 - i]])) / (dim + 1.f);
        }
    }
}

template <size_t dim>
__global__ void dedup(int* hash_entries, short* hash_keys, size_t hash_cap) {
    const size_t idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < hash_cap) {
        int& e = hash_entries[idx];
        if(e >= 0) {
            const short* key = &hash_keys[idx*dim];
            e = hash_lookup<dim>(hash_entries, hash_keys, hash_cap, key);
        }
    }
}

__global__ void find_valid(const int* hash_entries,
                           int* valid_entries,
                           int* n_valid,
                           size_t hash_cap) {

    const size_t idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    if (idx < hash_cap) {
        const int& e = hash_entries[idx];
        if(e >= 0) {
            const int my_ind = atomicAdd(n_valid, 1);
            valid_entries[my_ind] = e;
        }
    }
}

template<size_t dim>
void _call_hash_kernels(const float* points,
                int* hash_entries,
                short* hash_keys,
                int* neib_ents,
                float* barycentric,
                int* valid_entries,
                int* n_valid,
                size_t hash_cap,
                size_t N,
                cudaStream_t stream) {

    build_hash<dim><<<cuda_gridsize(N), BLOCK, 0, stream>>>(points,
            hash_entries, hash_keys, neib_ents, barycentric, hash_cap, N);
    dedup<dim><<<cuda_gridsize(hash_cap), BLOCK, 0, stream>>>(hash_entries, hash_keys, hash_cap);
    find_valid<<<cuda_gridsize(hash_cap), BLOCK, 0, stream>>>(hash_entries, valid_entries, n_valid, hash_cap);
}

void call_build_hash_kernels(const float* points,
                int* hash_entries,
                short* hash_keys,
                int* neib_ents,
                float* barycentric,
                int* valid_entries,
                int* n_valid,
                size_t hash_cap,
                size_t N, size_t dim,
                cudaStream_t stream) {

    switch(dim) {
        case 1:
            _call_hash_kernels<1>(points, hash_entries, hash_keys, neib_ents,
                    barycentric, valid_entries, n_valid, hash_cap, N, stream);
            break;
        case 2:
            _call_hash_kernels<2>(points, hash_entries, hash_keys, neib_ents,
                    barycentric, valid_entries, n_valid, hash_cap, N, stream);
            break;
        case 3:
            _call_hash_kernels<3>(points, hash_entries, hash_keys, neib_ents,
                    barycentric, valid_entries, n_valid, hash_cap, N, stream);
            break;
        case 4:
            _call_hash_kernels<4>(points, hash_entries, hash_keys, neib_ents,
                    barycentric, valid_entries, n_valid, hash_cap, N, stream);
            break;
        case 5:
            _call_hash_kernels<5>(points, hash_entries, hash_keys, neib_ents,
                    barycentric, valid_entries, n_valid, hash_cap, N, stream);
            break;
        default:
            fprintf(stderr,
            "Can't build hash tables for more than 5 dimensional points (but you can fix this by copy/pasting the above lines a few more times if you need to).\n");
            exit(-1);
    }
}
