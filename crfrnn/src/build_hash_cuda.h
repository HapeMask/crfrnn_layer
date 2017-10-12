int build_hash_cuda(const THCudaTensor* th_points,
        THCudaIntTensor* th_hash_entries,
        THCudaShortTensor* th_hash_keys,
        THCudaIntTensor* th_neib_ents,
        THCudaTensor* th_barycentric,
        THCudaIntTensor* th_valid_entries,
        THCudaIntTensor* n_valid,
        size_t hash_cap);
