import os
import glob
import platform
import paddle

from setuptools import find_packages
from setuptools import setup
from setuptools.extension import Extension

requirements = ["paddle"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))

    sources = main_file + source_cpu
    define_macros = []

    extra_compile_args = []
    if platform.system() == "Windows":
        extra_compile_args += ["/std:c++14"]
    else:
        extra_compile_args += ["-std=c++14"]

    ext_modules = [
        Extension(
            "MultiScaleDeformableAttention",
            sources=sources,
            include_dirs=[extensions_dir],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PaddlePaddle Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
)
