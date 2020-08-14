try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

#import distutils.sysconfig
from Cython.Build import cythonize
import numpy
import os
import sys

with open("src/geocat/ncomp/version.py") as f:
    exec(f.read())

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.normpath(sys.prefix)

include_dirs = [os.path.join(PREFIX, 'include'), numpy.get_include(),"."]

extensions = [
    Extension(
        "geocat.ncomp._ncomp",
        ["src/geocat/ncomp/_ncomp.pyx"],
        include_dirs=include_dirs,
        libraries=["ncomp"],
    ),
    Extension(
        "geocat.ncomp.carrayify",
        ["src/geocat/ncomp/carrayify.pyx"],
        include_dirs=include_dirs,
    )
]
setup(
    name="geocat.ncomp",
    ext_modules=cythonize(
        extensions,
        # help cythonize find my own .pxd files
        include_path=[os.path.join(SRC_DIR, "src/geocat/ncomp/_ncomp")]),
    package_dir={
        '': 'src',
        'geocat': 'src/geocat',
        'geocat.ncomp': 'src/geocat/ncomp'
    },
    package_data={'geocat': ['__init__.pxd', 'ncomp/*.pxd']},
    namespace_packages=['geocat'],
    packages=["geocat", "geocat.ncomp"],
    version=__version__,
    install_requires=[
        'numpy',
        'cython',
    ])
