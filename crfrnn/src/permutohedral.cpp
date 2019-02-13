#include <torch/extension.h>

#include "build_hash_cuda.h"
#include "gfilt_cuda.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gfilt_cuda", &gfilt_cuda, "High-dimensional Gaussian filter (CUDA)");
    m.def("build_hash_cuda", &build_hash_cuda, "High-dimensional Gaussian filter (CUDA)");
}
