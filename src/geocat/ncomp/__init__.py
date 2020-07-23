from . import _ncomp
# The following imports allow for the function name to be used directly under the package namespace, skipping the module name.
# This is done to maintain backwards compatibily from when the functions were defined in geocat/ncomp/__init__.py
from .dpres_plevel import dpres_plevel
from .eofunc import (eofunc, eofunc_ts)
from .errors import (Error, AttributeError, ChunkError, CoordinateError,
                     DimensionError, MetaError)
from .grid2triple import grid2triple
from .linint2 import linint2
from .linint2points import linint2_points
from .moc_globe_alt import moc_globe_atl
from .rcm2points import rcm2points
from .rcm2rgrid import rcm2rgrid
from .rgrid2rcm import rgrid2rcm
from .triple2grid import triple2grid
from .version import __version__
