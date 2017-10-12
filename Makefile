.PHONY: all clean

all: build/hash_kernels.o build/gfilt_kernels.o

PYTORCH_BASE=$(shell python -c "import os; import torch; print(os.path.dirname(os.path.realpath(torch.__file__)))")
TORCH_INCLUDE="$(PYTORCH_BASE)/lib/include"

build/hash_kernels.o: crfrnn/src/build_hash.cu crfrnn/src/hash_fns.cuh
	-mkdir build
	nvcc $(NVCC_FLAGS) -Xcompiler=-fPIC -c crfrnn/src/build_hash.cu -o build/hash_kernels.o

build/gfilt_kernels.o: crfrnn/src/gfilt_cuda.cu crfrnn/src/hash_fns.cuh
	-mkdir build
	nvcc $(NVCC_FLAGS) -Xcompiler=-fPIC -c crfrnn/src/gfilt_cuda.cu -o build/gfilt_kernels.o

clean:
	-rm -r build/*
