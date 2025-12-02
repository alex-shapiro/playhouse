"""Build script for playhouse C++/CUDA extensions."""

import os

from setuptools import setup
from torch.utils.cpp_extension import (
    CUDA_HOME,
    ROCM_HOME,
    BuildExtension,
    CppExtension,
    CUDAExtension,
)

# Build CUDA extension if torch can find CUDA or HIP/ROCM
BUILD_CUDA = bool(CUDA_HOME or ROCM_HOME)

DEBUG = os.getenv("DEBUG", "0") == "1"

# Compile flags
cxx_args = ["-O3"]
nvcc_args = ["-O3"]

if DEBUG:
    cxx_args = ["-O0", "-g"]
    nvcc_args = ["-O0", "-g"]

# Extension sources
sources = ["playhouse/extensions/gae.cpp"]
if BUILD_CUDA:
    sources.append("playhouse/extensions/cuda/gae.cu")
    Extension = CUDAExtension
else:
    Extension = CppExtension

ext_modules = [
    Extension(
        "playhouse._C",
        sources,
        extra_compile_args={
            "cxx": cxx_args,
            "nvcc": nvcc_args,
        },
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
