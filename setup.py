import os

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

GFILT_CALL = "_call_gfilt_kernels<{ref_dim}, {val_dim}>(values, output, tmp_vals_1, tmp_vals_2, hash_entries, hash_keys, neib_ents, barycentric, valid_entries, n_valid, hash_cap, N, reverse, stream);"
def make_gfilt_dispatch_table(fname, ref_dims=range(2, 6), val_dims=range(1, 16)):
    with open(fname, "w") as f:
        f.write("switch(1000 * ref_dim + val_dim) {\n")
        for rdim in ref_dims:
            for vdim in val_dims:
                f.write(f"\tcase {1000 * rdim + vdim}:\n")
                f.write("\t\t" + GFILT_CALL.format(ref_dim=rdim, val_dim=vdim) + "\n")
                f.write("\t\tbreak;\n")
        f.write("\tdefault:\n")
        f.write("\t\tprintf(\"Unsupported ref_dim/val_dim combination (%zd, %zd), generate a new dispatch table using 'make_gfilt_dispatch.py'.\\n\", ref_dim, val_dim);\n")
        f.write("\t\texit(-1);\n")
        f.write("}\n")

if __name__ == "__main__":
    table_fn = "src/gfilt_dispatch_table.h"
    if not os.path.exists(table_fn) or (os.path.getmtime(table_fn) < os.path.getmtime(__file__)):
        make_gfilt_dispatch_table(table_fn)

    cxx_args = ["-O3", "-fopenmp", "-std=c++14"]
    nvcc_args = ["-O3"]
    if "CC" in os.environ:
        nvcc_args.append("-ccbin=" + os.path.dirname(os.environ.get("CC")))

    setup(
        name="permutohedral",
        version="0.4",
        description="",
        url="",
        author="Gabriel Schwartz",
        author_email="gbschwartz@gmail.com",
        ext_modules=[
            CUDAExtension(
                "permutohedral_ext",
                [
                    "src/permutohedral.cpp",
                    "src/build_hash_wrapper.cu",
                    "src/gfilt_wrapper.cu",
                    "src/gfilt_cuda.cu",
                    "src/build_hash.cu"
                ],
                extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
            )
        ],
        cmdclass={"build_ext": BuildExtension},
        packages=["permutohedral"],
    )

    setup(name="crfrnn", packages=["crfrnn"])
