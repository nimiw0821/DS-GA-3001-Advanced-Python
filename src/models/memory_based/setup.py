### cython.pyx?
#%load_ext Cython
#%%cython
from distutils.core import setup
import Cython
from Cython.Build import cythonize

setup(
    name = "models_cython",
    ext_modules = cythonize("*.pyx"), 
)
#python setup.py build_ext --inplace