geocat-ncomp
===========

GeoCAT-ncomp is the computational component of the [GeoCAT](https://ncar.github.io/GeoCAT) project. GeoCAT-ncomp wraps NCL's non-WRF Fortran routines into Python.

GeoCAT-ncomp depends on a separate C/Fortran library called "[ncomp](https://github.com/NCAR/ncomp)", which contains these Fortran routines.


Documentation
=============

[GeoCAT-ncomp documentation on Read the Docs](https://geocat-ncomp.readthedocs.io)



Installation and build instructions
===================================

Please see our documentation for [installation instructions](https://geocat-ncomp.readthedocs.io/en/latest/installation.html) and [build instructions](https://geocat-ncomp.readthedocs.io/en/latest/installation.html#building-geocat-ncomp-from-source).


Xarray interface vs NumPy interface
===================================

GeoCAT-ncomp provides a high-level Xarray interface under the `geocat.ncomp` namespace. However, a stripped-down NumPy interface is used under the hood to bridge the gap between NumPy arrays and the C data structures used by `NComp`. These functions are accessible under the `geocat.ncomp._ncomp` namespace, but are minimally documented and are intended primarily for internal use.
