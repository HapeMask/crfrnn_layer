__device__ int DIM_SPECIFIC(hash_insert)(int* entries, short* keys, size_t capacity, const short* key) {
    unsigned int h = DIM_SPECIFIC(hash)(key) % capacity;
    const unsigned int init_h = h;
    const size_t max_it = min((size_t)MIN_QUAD_PROBES, capacity);
    int* entry = &entries[h];

    // First try quadratic probing for a fixed number of iterations, then fall
    // back to linear probing.
    for(int iter=0; iter<max_it; ++iter) {
        entry = &entries[h];
        const int ret = atomicCAS(entry, -1, -2);

        if (ret == -1) {
            // This thread got the lock and it was empty, fill it.
            *(entry) = h;
            for(int i=0; i<KEY_DIM; ++i) { keys[h*KEY_DIM + i] = key[i]; }
            return h;
        } else if (ret >= 0 && DIM_SPECIFIC(key_cmp)(&keys[ret*KEY_DIM], key)){
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
            for(int i=0; i<KEY_DIM; ++i) { keys[h*KEY_DIM + i] = key[i]; }
            return h;
        } else if (ret >= 0 && DIM_SPECIFIC(key_cmp)(&keys[ret*KEY_DIM], key)){
            // If another thread already inserted the same key, return the
            // entry for that key.
            return ret;
        }

        h = (h+1) % capacity;
    }

    // We wrapped around without finding a free slot, the table is full.
    return -1;
}

__device__ __inline__ void DIM_SPECIFIC(embed)(const float* f, float* e, size_t N) {
    e[REF_DIM] = -sqrt(DR/(DR+1.f)) * f[N*(REF_DIM-1)];
    for(int i=REF_DIM-1; i>0; --i) {
        e[i] = f[N*i]/sqrt((i+1.f)/(i+2.f)) + e[i+1] - sqrt(i/(i+1.f)) * f[N*(i-1)];
    }
    e[0] = f[0] / sqrt(0.5f) + e[1];
}

__device__ __inline__ short DIM_SPECIFIC(round2mult)(float f) {
    const float s = f / (DR+1.f);
    const float lo = floor(s) * (DR+1.f);
    const float hi = ceil(s) * (DR+1.f);
    return ((hi-f) > (f-lo)) ? lo : hi;
}

__device__ __inline__ void DIM_SPECIFIC(ranksort_diff)(const float* v1, const short* v2, short* rank) {
    for(int i=0; i<REF_DIM+1; ++i) {
        rank[i] = 0;
        const float di = v1[i] - v2[i];
        for(int j=0; j<REF_DIM+1; ++j) {
            const float dj = v1[j] - v2[j];
            if (di < dj || (di==dj && i>j)) { ++rank[i]; }
        }
    }
}

__device__ __inline__ short DIM_SPECIFIC(canonical_coord)(int k, int i) {
    return (i < (REF_DIM+1-k)) ?  k : (k - (REF_DIM+1));
}

KERNEL void DIM_SPECIFIC(build_hash)(GLOBAL_MEM const float* _points, size_t points_off,
                       GLOBAL_MEM int* _hash_entries, size_t entries_off,
                       GLOBAL_MEM short* _hash_keys, size_t keys_off,
                       GLOBAL_MEM int* _neib_ents, size_t neib_ents_off,
                       GLOBAL_MEM float* _barycentric, size_t bary_off,
                       size_t hash_cap,
                       size_t N) {

    const float* points = _points + points_off;
    int* hash_entries = _hash_entries + entries_off;
    short* hash_keys = _hash_keys + keys_off;
    int* neib_ents = _neib_ents + neib_ents_off;
    float* barycentric = _barycentric + bary_off;

    const size_t idx = GID_0 * LDIM_0 + LID_0;

    // The temporary key has 1 extra dimension. Normally we ignore the last
    // entry in the key because they sum to 0, but we can use the key as swap
    // space to invert the sorting permutation.
    short tmpkey[REF_DIM+1];
    short rounded[REF_DIM+1];
    short rank[REF_DIM+1];
    float embedded[REF_DIM+1];

    if (idx < N) {
        const float* p = &points[idx];
        float* b = &barycentric[idx];

        DIM_SPECIFIC(embed)(p, embedded, N);

        short sum = 0;
        for(int i=0; i<REF_DIM+1; ++i) {
            const short r = DIM_SPECIFIC(round2mult)(embedded[i]);
            rounded[i] = r;
            sum += r;
        }
        sum /= (REF_DIM+1);

        // Compute rank(embedded - rounded), decreasing order
        DIM_SPECIFIC(ranksort_diff)(embedded, rounded, rank);

        // Walk the point back onto H_d (Lemma 2.9 in permutohedral_techreport.pdf)
        if (sum > 0) {
            for(int i=0; i<REF_DIM+1; ++i) {
                if(rank[i] >= REF_DIM+1-sum) { rounded[i] -= REF_DIM+1; }
            }
        } else if (sum < 0) {
            for(int i=0; i<REF_DIM+1; ++i) {
                if(rank[i] < -sum) { rounded[i] += REF_DIM+1; }
            }
        }

        // Re-ompute rank(embedded - rounded), decreasing order. TODO: The
        // reference code doesn't do this and does some other things instead.
        // Find out why?
        DIM_SPECIFIC(ranksort_diff)(embedded, rounded, rank);

        for(int k=0; k<REF_DIM+1; ++k) {
            for(int i=0; i<KEY_DIM; ++i) {
                tmpkey[i] = DIM_SPECIFIC(canonical_coord)(k, rank[i]) + rounded[i];
            }

            const int ind = DIM_SPECIFIC(hash_insert)(hash_entries, hash_keys, hash_cap, tmpkey);
            assert(ind >= 0);
            neib_ents[idx + N*k] = ind;
        }

        // Rank is equivalent to the inverse permutation for sorting in
        // decreasing order, invert it here to compute barycentric coordinates
        // (which requires the non-inverted permutation).
        for(int i=0; i<REF_DIM+1; ++i) { tmpkey[rank[i]] = i; }
        for(int i=0; i<REF_DIM+1; ++i) { rank[i] = tmpkey[i]; }

        b[0] = 1 - ((embedded[rank[0]] - rounded[rank[0]]) -
                    (embedded[rank[REF_DIM]] - rounded[rank[REF_DIM]])) / (DR+1.f);
        for(int i=1; i<REF_DIM+1; ++i) {
            b[N*i] = ((embedded[rank[REF_DIM-i]]-rounded[rank[REF_DIM-i]]) -
                    (embedded[rank[REF_DIM+1-i]]-rounded[rank[REF_DIM+1-i]])) / (DR+1.f);
        }
    }
}
