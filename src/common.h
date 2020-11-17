#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#ifdef TORCH_CHECK
#define MYCHECK TORCH_CHECK
#else
#define MYCHECK AT_ASSERTM
#endif

#define CHECK_CUDA(x) MYCHECK(x.device().is_cuda(), #x " is not a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) MYCHECK(x.is_contiguous(), #x " is not contiguous.")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)
#define CHECK_2DIMS(x) MYCHECK((x.sizes().size() == 2), #x " is not 2 dimensional.")
#define CHECK_3DIMS(x) MYCHECK((x.sizes().size() == 3), #x " is not 3 dimensional.")
#define CHECK_4DIMS(x) MYCHECK((x.sizes().size() == 4), #x " is not 4 dimensional.")

#ifdef USE_OLD_PYTORCH_DATA_MEMBER
#define DATA_PTR(x,t) x.data<t>()
#else
#define DATA_PTR(x,t) x.data_ptr<t>()
#endif

