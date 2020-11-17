#include <torch/extension.h>
#include <torch/script.h>

#include "build_hash_cuda.h"
#include "gfilt_cuda.h"


PYBIND11_MODULE(permutohedral_ext, m) {
    m.def("gfilt_cuda", &gfilt_cuda, "High-dimensional Gaussian filter (CUDA)");
    m.def("build_hash_cuda", &build_hash_cuda, "High-dimensional Gaussian filter (CUDA)");
}

TORCH_LIBRARY(permutohedral_ext, m) {
    m.def("gfilt_cuda", &gfilt_cuda);
    m.def("build_hash_cuda", &build_hash_cuda);
}
