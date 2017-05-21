KERNEL void DIM_SPECIFIC(dedup)(GLOBAL_MEM int* _hash_entries, size_t entries_off,
                                GLOBAL_MEM short* _hash_keys, size_t keys_off,
                                GLOBAL_MEM size_t hash_cap) {

    int* hash_entries = _hash_entries + entries_off;
    short* hash_keys = _hash_keys + keys_off;

    const size_t idx = GID_0 * LDIM_0 + LID_0;

    if (idx < hash_cap ) {
        int& e = hash_entries[idx];
        if(e >= 0) {
            const short* key = &hash_keys[idx*KEY_DIM];
            e = hash_lookup(hash_entries, hash_keys, hash_cap, key);
        }
    }
}
