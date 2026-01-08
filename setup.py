import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '0.1.0'
sources = ["src/pybind.cpp","src/tensor.cu","src/conv.cu","src/pooling.cu","src/linear.cu","src/functional.cu"]

setup(
    name="mytensor",
    version=__version__,
    # author="LiYixin",
    author_email="2200013104@stu.pku.edu.cn",
    py_modules=["mytensor"],
    packages=find_packages(exclude=("tests",)),
    zip_safe=False,
    install_requires=[
        "torch",],
    ext_modules=[
        CUDAExtension(name="mytensor",
                      sources=sources,
                      libraries=["cublas"], 
                      )],
    cmdclass={"build_ext": BuildExtension},
)
