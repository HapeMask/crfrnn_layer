import sys

base_call = "_call_gfilt_kernels<%d, %d>(values, output, tmp_vals_1, tmp_vals_2, hash_entries, hash_keys, neib_ents, barycentric, valid_entries, n_valid, hash_cap, N, reverse, stream);"

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2:
        max_ref_dim, max_val_dim = args
        ref_dims = range(1, max_ref_dim+1)
        val_dims = range(1, max_val_dim+1)
    else:
        assert(len(args) == 0)
        ref_dims = [3, 6]
        val_dims = range(1, 16)

    print("switch(1000 * ref_dim + val_dim) {")
    for rdim in ref_dims:
        for vdim in val_dims:
            print("\tcase %d:" % (1000 * rdim + vdim))
            print("\t\t" + (base_call % (rdim, vdim)))
            print("\t\tbreak;")
    print("\tdefault:")
    print("\t\tprintf(\"Unsupported ref_dim/val_dim combination (%zd, %zd), generate a new dispatch table using 'make_gfilt_dispatch.py'.\\n\", ref_dim, val_dim);")
    print("\t\texit(-1);")
    print("}")
