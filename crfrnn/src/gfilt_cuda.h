int gfilt_cuda(const THCudaTensor* th_values, THCudaTensor* th_output,
        THCudaTensor* th_tmp_vals_1, THCudaTensor* th_tmp_vals_2,
        THCudaIntTensor* th_hash_entries, THCudaShortTensor* th_hash_keys,
        THCudaIntTensor* th_neib_ents, THCudaTensor* th_barycentric,
        THCudaIntTensor* th_valid_entries, int n_valid,
        size_t hash_cap, size_t ref_dim);
