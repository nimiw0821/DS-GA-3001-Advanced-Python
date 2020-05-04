### cython.pyx?
#%load_ext Cython
#%%cython
from distutils.core import setup
import Cython
from Cython.Build import cythonize

setup(
    name = "model3_cython",
    ext_modules = cythonize("model3.pyx"), 
)
#python setup.py build_ext --inplace