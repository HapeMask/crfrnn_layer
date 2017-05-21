KERNEL void DIM_SPECIFIC(splat)(GLOBAL_MEM const float* _values, size_t values_off,
                                GLOBAL_MEM const float* _barycentric, size_t bary_off,
                                GLOBAL_MEM const int* _hash_entries, size_t entries_off,
                                GLOBAL_MEM const int* _neib_ents, size_t neib_ents_off,
                                GLOBAL_MEM float* hash_values, size_t N) {

    const float* values = _values + values_off;
    const float* barycentric = _barycentric + bary_off;
    const int* hash_entries = _hash_entries + entries_off;
    const int* neib_ents = _neib_ents + neib_ents_off;

    const size_t idx = GID_0 * LDIM_0 + LID_0;
    float local_v[VAL_DIM];
    float local_b[REF_DIM+1];

    if (idx < N) {
        for(int i=0; i<VAL_DIM; ++i) { local_v[i] = values[idx + N*i]; }
        for(int i=0; i<(REF_DIM+1); ++i) { local_b[i] = barycentric[idx + N*i]; }

        // Splat this point onto each vertex of its surrounding simplex.
        for(int k=0; k<REF_DIM+1; ++k) {
            const int& ind = hash_entries[neib_ents[idx + N*k]];
            float* hv = &hash_values[ind*VAL_DIM];

            for(int i=0; i<VAL_DIM; ++i) {
                atomicAdd(&hv[i], local_b[k] * local_v[i]);
            }
        }
    }
}
