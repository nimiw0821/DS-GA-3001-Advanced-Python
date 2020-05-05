from distutils.core import setup,Extension
from Cython.Build import cythonize
import numpy


setup(
    name = 'als cython',
    ext_modules = cythonize('/Users/zihaoguo/NYU/ADPY/DS-GA-3001-Advanced-Python/src/models/als/model_cython.pyx'),
    include_dirs=[numpy.get_include()]
)
#include_dirs = [numpy.get_include()]
