from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name="crfrnn",
      version="0.2",
      description="",
      url="",
      author="Gabriel Schwartz",
      author_email="gbschwartz@gmail.com",
      ext_modules=[
          CUDAExtension("crfrnn",
              [ "crfrnn/src/permutohedral.cpp",
                "crfrnn/src/build_hash_wrapper.cu",
                "crfrnn/src/gfilt_wrapper.cu",
                "crfrnn/src/gfilt_cuda.cu",
                "crfrnn/src/build_hash.cu",
                ],
              extra_compile_args={"cxx": ["-D_GLIBCXX_USE_CXX11_ABI=1", "-Wl,--no-as-needed"], "nvcc": ["-D_GLIBCXX_USE_CXX11_ABI=1"]},
              extra_link_args=["-Wl,--no-as-needed"],
              )
      ],
      cmdclass = {
          "build_ext": BuildExtension
      }
)
