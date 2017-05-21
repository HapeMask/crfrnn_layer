KERNEL void DIM_SPECIFIC(slice)(GLOBAL_MEM float* _out, size_t out_off,
                                GLOBAL_MEM const float* _barycentric, size_t bary_off,
                                GLOBAL_MEM const int* _hash_entries, size_t entries_off,
                                GLOBAL_MEM const int* _neib_ents, size_t neib_ent_off,
                                GLOBAL_MEM const float* hash_values, int N) {

    float* out = _out + out_off;
    const float* barycentric = _barycentric + bary_off;
    const int* hash_entries = _hash_entries + entries_off;
    const int* neib_ents = _neib_ents + neib_ent_off;

    const size_t idx = GID_0 * LDIM_0 + LID_0;

    float local_out[VAL_DIM];

    if (idx < N) {
        for(int i=0; i<VAL_DIM; ++i) { local_out[i] = 0; }

        // Gather values from each of the surrounding simplex vertices.
        for(int k=0; k<REF_DIM+1; ++k) {
            const int& ind = hash_entries[neib_ents[idx + N*k]];
            const float* hv = &hash_values[ind*VAL_DIM];

            for(int i=0; i<VAL_DIM; ++i) {
                local_out[i] += barycentric[idx + N*k] * hv[i];
            }
        }

        for(int i=0; i<VAL_DIM; ++i) {
            out[idx + N*i] = local_out[i];
        }
    }
}
