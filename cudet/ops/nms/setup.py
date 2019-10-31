import os.path as osp
from setuptools import setup,Extension

import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from torch.utils.cpp_extension import BuildExtension,CUDAExtension