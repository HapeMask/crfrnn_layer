import os
import sys

from setuptools import setup, find_packages
import build

base_dir = os.path.dirname(os.path.realpath(__file__))

setup(name="crfrnn",
      version="0.1",
      description="",
      url="",
      author="Gabriel Schwartz",
      author_email="hapemask@gmail.com",
      install_requires=["cffi>=1.0.0"],
      setup_requires=["cffi>=1.0.0"],
      packages=find_packages(exclude=["build"]),
      ext_package="",
      cffi_modules=[os.path.join(base_dir, "build.py:ffi")]
)
