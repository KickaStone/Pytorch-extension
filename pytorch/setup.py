import os
import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# set path as running directory
include_dir = 'include'
kernel_dir = 'kernel'
pytorch_dir = 'pytorch'

sources = glob.glob(os.path.join(kernel_dir, '*.cu')) + glob.glob(os.path.join(pytorch_dir, '*.cpp'))
headers = glob.glob(os.path.join(include_dir, '*.h'))

setup(
    name="MyReLU",
    version="0.1",
    ext_modules=[
        CUDAExtension(
            "MyReLU",
            sources=sources,
            include_dirs=[include_dir]
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)