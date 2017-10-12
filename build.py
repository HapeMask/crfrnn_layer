import os
import torch
from torch.utils.ffi import create_extension

sources = []
headers = []
defines = []
extra_objects = []
include_dirs=[]
library_dirs=[]
libraries=[]
#sources = ["crfrnn/src/hash.cpp"]
#headers = ["crfrnn/src/hash.hpp"]
with_cuda = False

base_path = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available():
    sources += ["crfrnn/src/build_hash_cuda.c", "crfrnn/src/gfilt_cuda.c"]
    headers += ["crfrnn/src/build_hash_cuda.h", "crfrnn/src/gfilt_cuda.h"]
    defines += [("WITH_CUDA", None)]
    with_cuda = True
    library_dirs += ["/opt/cuda/lib64"]
    include_dirs += ["/opt/cuda/include"]
    libraries += ["cuda", "cudart"]

    extra_objects = [os.path.join(base_path, "build/hash_kernels.o")]
    extra_objects += [os.path.join(base_path, "build/gfilt_kernels.o")]

ffi = create_extension("crfrnn._ext.permutohedral",
        package=True,
        headers=headers,
        sources=sources,
        define_macros=defines,
        relative_to=__file__,
        with_cuda=with_cuda,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_objects=extra_objects
        )

if __name__ == "__main__":
    ffi.build()
