#include "hash_fns.cuh"
#include "build_hash_kernel.h"

dim3 cuda_gridsize(int n) {
    int k = (n - 1) / BLOCK + 1;
    int x = k;
    int y = 1;

    if(x > 65535) {
        x = ceil(sqrt(k));
        y = (n - 1) / (x * BLOCK) + 1;
    }

    return dim3(x, y, 1);
}

template <size_t key_dim>
__device__ unsigned int hash(const short* key) {
    unsigned int h = 0;
    for (int i=0; i < key_dim; ++i) {
        h ^= ((unsigned int)key[i]) << ((31/key_dim)*i);
    }
    return h;
}

template <size_t key_dim>
__device__ bool key_cmp(const short* key1, const short* key2) {
    for(int i=0; i<key_dim; ++i) {
        if (key1[i] != key2[i]) { return false; }
    }
    return true;
}

template <size_t key_dim>
__device__ int hash_lookup(const int* entries, const short* keys, size_t capacity, const short* key) {
    unsigned int h = hash<key_dim>(key) % capacity;
    const unsigned int init_h = h;
    const size_t max_it = min((size_t)MIN_QUAD_PROBES, capacity);

    // The probing sequence here needs to match the one used for insertion,
    // otherwise bad things will happen.
    for(int iter=0; iter<max_it; ++iter) {
        const int& entry = entries[h];
        if (entry == -1 || key_cmp<key_dim>(&keys[entry*key_dim], key)) { return entry; }
        h = (init_h + iter*iter) % capacity;
    }

    for(int iter=0; iter<capacity; ++iter) {
        const int& entry = entries[h];
        if (entry == -1 || key_cmp<key_dim>(&keys[entry*key_dim], key)) { return entry; }
        h = (h+1) % capacity;
    }

    // We wrapped around without finding a matching key, the key is not in the
    // table.
    return -1;
}
