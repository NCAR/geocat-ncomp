GeoCAT-comp is both the whole computational component of the [GeoCAT](https://ncar.github.io/GeoCAT) 
project and a single Github repository as described in [GeoCAT-comp](https://github.com/NCAR/geocat-comp). 
As the computational component of [GeoCAT](https://ncar.github.io/GeoCAT), GeoCAT-comp provides implementations of 
computational functions for operating on geosciences data. Many of these functions originated in NCL are pivoted into 
Python with the help of GeoCAT-comp; however, developers are welcome to come up with novel computational functions 
for geosciences data.

Many of the computational functions under GeoCAT-comp are implemented in Fortran 
(or possibly C). However, others can be implemented in a pure Python fashion. To facilitate 
contribution, the whole GeoCAT-comp computational component is split into three Github repositories with respect to 
being pure-Python, Python with Cython wrappers for compiled codes, and compiled language (C and Fortran) 
implementations. Such implementation layers are handled within [GeoCAT-comp](https://github.com/NCAR/geocat-comp), 
GeoCAT-ncomp (this repository), and [libncomp](https://github.com/NCAR/libncomp) 
repositories, respectively (GeoCAT-comp and libncomp repos are documented on their own).


# GeoCAT-ncomp

GeoCAT-ncomp wraps, in Cython, the compiled language implementations of functions found in the 
[libncomp](https://github.com/NCAR/libncomp) repository. Developers basing their implementations entirely in Python need 
not concern themselves with this repo; instead, they should engage with 
[GeoCAT-comp](https://github.com/NCAR/geocat-comp) repo as it invisibly imports GeoCAT-ncomp. However, for those 
functions that are implemented in Fortran (or possibly C or C++), this repo provides a Python interface to those 
functions via a Cython wrapper.


# Documentation

[GeoCAT Homepage](https://geocat.ucar.edu/)

[GeoCAT Contributor's Guide](https://geocat.ucar.edu/pages/contributing.html)

[GeoCAT-comp documentation on Read the Docs](https://geocat-comp.readthedocs.io)


# Installation and build instructions

Please see our documentation for 
[installation and build instructions](https://github.com/NCAR/geocat-ncomp/INSTALLATION.md).


# Xarray interface vs NumPy interface

GeoCAT-ncomp provides a high-level Xarray interface under the `geocat.ncomp` namespace. However, 
a stripped-down NumPy interface is used under the hood to bridge the gap between NumPy arrays and 
the C data structures used by `libncomp`. These functions are accessible under the `geocat.comp._ncomp` namespace, 
but are minimally documented and are intended primarily for internal use.
