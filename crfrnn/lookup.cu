__device__ int hash_lookup(const int* entries, const short* keys, size_t capacity, const short* key) {
    unsigned int h = DIM_SPECIFIC(hash)(key) % capacity;
    const unsigned int init_h = h;
    const size_t max_it = min((size_t)MIN_QUAD_PROBES, capacity);

    // The probing sequence here needs to match the one used for insertion,
    // otherwise bad things will happen.
    for(int iter=0; iter<max_it; ++iter) {
        const int& entry = entries[h];
        if (entry == -1 || DIM_SPECIFIC(key_cmp)(&keys[entry*KEY_DIM], key)) { return entry; }
        h = (init_h + iter*iter) % capacity;
    }

    for(int iter=0; iter<capacity; ++iter) {
        const int& entry = entries[h];
        if (entry == -1 || DIM_SPECIFIC(key_cmp)(&keys[entry*KEY_DIM], key)) { return entry; }
        h = (h+1) % capacity;
    }

    // We wrapped around without finding a matching key, the key is not in the
    // table.
    return -1;
}
