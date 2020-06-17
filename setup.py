try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

import os
import sys

import numpy
# import distutils.sysconfig
from Cython.Build import cythonize

with open("src/geocat/ncomp/version.py") as f:
    exec(f.read())

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.normpath(sys.prefix)

include_dirs = [os.path.join(PREFIX, 'include'), numpy.get_include()]

extensions = [
    Extension("geocat.ncomp._ncomp", ["src/geocat/ncomp/_ncomp.pyx"],
              include_dirs=include_dirs,
              libraries=["ncomp"],
              ),
]
setup(
    name="geocat.ncomp",
    ext_modules=cythonize(extensions,
                          # help cythonize find my own .pxd files
                          include_path=[os.path.join(SRC_DIR, "src/geocat/ncomp/_ncomp")]),
    package_dir={'': 'src', 'geocat': 'src/geocat', 'geocat.ncomp': 'src/geocat/ncomp', 'geocat.ntest': 'test'},
    package_data={'geocat': ['__init__.pxd', 'ncomp/*.pxd']},
    namespace_packages=['geocat'],
    packages=['geocat', 'geocat.ncomp', 'geocat.ntest'],
    version=__version__,
    install_requires=[
        'numpy',
        'cython',
    ]
)
