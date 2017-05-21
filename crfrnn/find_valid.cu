KERNEL void DIM_SPECIFIC(find_valid)(GLOBAL_MEM const int* _hash_entries, size_t entries_off,
                                          GLOBAL_MEM int* _valid_entries, size_t valid_entries_off,
                                          GLOBAL_MEM int* _n_valid, size_t n_valid_off,
                                          GLOBAL_MEM size_t hash_cap) {
    const int* hash_entries = _hash_entries + entries_off;
    int* valid_entries = _valid_entries + valid_entries_off;
    int* n_valid = _n_valid + n_valid_off;

    const size_t idx = GID_0 * LDIM_0 + LID_0;

    if (idx < hash_cap ) {
        const int& e = hash_entries[idx];
        if(e >= 0) {
            const int my_ind = atomicAdd(n_valid, 1);
            valid_entries[my_ind] = e;
        }
    }
}
