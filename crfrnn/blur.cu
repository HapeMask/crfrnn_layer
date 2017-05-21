KERNEL void DIM_SPECIFIC(blur)(GLOBAL_MEM float* out,
                               GLOBAL_MEM const int* _hash_entries, size_t entries_off,
                               GLOBAL_MEM const int* _valid_entries, size_t valid_ent_off,
                               GLOBAL_MEM const short* _hash_keys, size_t keys_off,
                               GLOBAL_MEM const float* hash_values,
                               size_t hash_cap, size_t n_valid, size_t axis) {

    const int* hash_entries = _hash_entries + entries_off;
    const int* valid_entries = _valid_entries + valid_ent_off;
    const short* hash_keys = _hash_keys + keys_off;

    const size_t idx = GID_0 * LDIM_0 + LID_0;

    float local_out[VAL_DIM];
    short key[KEY_DIM];

    if (idx < n_valid ) {
        const int& ind_c = valid_entries[idx];
        for(int i=0; i<VAL_DIM; ++i) { local_out[i] = 0; }

        const short* key_c = &hash_keys[ind_c*KEY_DIM];

        for(int i=0; i<KEY_DIM; ++i) { key[i] = key_c[i] + 1; }
        key[axis] = key_c[axis] - REF_DIM;
        const int ind_l = hash_lookup(hash_entries, hash_keys, hash_cap, key);

        for(int i=0; i<KEY_DIM; ++i) { key[i] = key_c[i] - 1; }
        key[axis] = key_c[axis] + REF_DIM;
        const int ind_r = hash_lookup(hash_entries, hash_keys, hash_cap, key);

        if(ind_l >= 0 && ind_r >= 0) {
            for(int i=0; i<VAL_DIM; ++i) {
                local_out[i] = (hash_values[ind_l*VAL_DIM + i] +
                         2*hash_values[ind_c*VAL_DIM + i] +
                         hash_values[ind_r*VAL_DIM + i]) * 0.25f;
            }
        } else if(ind_l >= 0) {
            for(int i=0; i<VAL_DIM; ++i) {
                local_out[i] = (hash_values[ind_l*VAL_DIM + i] +
                         2*hash_values[ind_c*VAL_DIM + i]) * 0.25f;
            }
        } else if(ind_r >= 0) {
            for(int i=0; i<VAL_DIM; ++i) {
                local_out[i] = (hash_values[ind_r*VAL_DIM + i] +
                         2*hash_values[ind_c*VAL_DIM + i]) * 0.25f;
            }
        } else {
            for(int i=0; i<VAL_DIM; ++i) {
                local_out[i] = hash_values[ind_c*VAL_DIM + i] * 0.5f;
            }
        }

        for(int i=0; i<VAL_DIM; ++i) { out[ind_c*VAL_DIM + i] = local_out[i]; }
    }
}
