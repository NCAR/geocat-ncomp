import numpy as np
import xarray as xr

from . import _ncomp
# The following imports allow for the function name to be used directly under the package namespace, skipping the module name.
# This is done to maintain backwards compatibily from when the functions were defined in geocat/ncomp/__init__.py
from .errors import (
    DimensionError, MetaError)


def triple2grid(x, y, data, xgrid, ygrid, **kwargs):
    """Places unstructured (randomly-spaced) data onto the nearest locations of a rectilinear grid.

    Args:

	x (:class:`numpy.ndarray`):
            One-dimensional arrays of the same length containing the coordinates
            associated with the data values. For geophysical variables, x
            correspond to longitude.

	y (:class:`numpy.ndarray`):
            One-dimensional arrays of the same length containing the coordinates
            associated with the data values. For geophysical variables, y
            correspond to latitude.

	data (:class:`numpy.ndarray`):
            A multi-dimensional array, whose rightmost dimension is the same
            length as `x` and `y`, containing the values associated with the `x`
            and `y` coordinates. Missing values, may be present but will be ignored.

	xgrid (:class:`numpy.ndarray`):
            A one-dimensional array of length M containing the `x` coordinates
            associated with the returned two-dimensional grid. For geophysical
            variables, these are longitudes. The coordinates' values must be
            monotonically increasing.

	ygrid (:class:`numpy.ndarray`):
            A one-dimensional array of length N containing the `y` coordinates
            associated with the returned two-dimensional grid. For geophysical
            variables, these are latitudes. The coordinates' values must be
            monotonically increasing.

        **kwargs:
            extra options for the function. Currently the following are supported:
            - ``method``: An integer value that defaults to 1 if option is True,
                          and 0 otherwise. A value of 1 means to use the great
                          circle distance formula for distance calculations.
            - ``domain``: A float value that should be set to a value >= 0. The
                          default is 1.0. If present, the larger this factor the
                          wider the spatial domain allowed to influence grid boundary
                          points. Typically, `domain` is 1.0 or 2.0. If `domain` <= 0.0,
                          then values located outside the grid domain specified by
                          `xgrid` and `ygrid` arguments will not be used.
            - ``distmx``: Setting option@distmx allows the user to specify a search
                          radius (km) beyond which observations are not considered
                          for nearest neighbor. Only applicable when `method` = 1.
                          The default `distmx`=1e20 (km) means that every grid point
                          will have a nearest neighbor. It is suggested that users
                          specify some reasonable value for distmx.
            - ``msg`` (:obj:`numpy.number`): A numpy scalar value that represent
                          a missing value in `data`. This argument allows a user to
                          use a missing value scheme other than NaN or masked arrays,
                          similar to what NCL allows.
            - ``meta`` (:obj:`bool`): If set to True and the input array is an Xarray,
                          the metadata from the input array will be copied to the
                          output array; default is False.
                          Warning: this option is not currently supported.

    Returns:
	:class:`numpy.ndarray`: The return array will be K x N x M, where K
        represents the leftmost dimensions of data. It will be of type double if
        any of the input is double, and float otherwise.

    Description:
        This function puts unstructured data (randomly-spaced) onto the nearest
        locations of a rectilinear grid. A default value of `domain` option is
        now set to 1.0 instead of 0.0.

        This function does not perform interpolation; rather, each individual
        data point is assigned to the nearest grid point. It is possible that
        upon return, grid will contain grid points set to missing value if
        no `x(n)`, `y(n)` are nearby.

    Examples:

	Example 1: Using triple2grid with :class:`xarray.DataArray` input

	.. code-block:: python

	    import numpy as np
	    import xarray as xr
	    import geocat.comp

	    # Open a netCDF data file using xarray default engine and load the data stream
	    ds = xr.open_dataset("./ruc.nc")

	    # [INPUT] Grid & data info on the source curvilinear
	    data = ds.DIST_236_CBL[:]
	    x = ds.gridlat_236[:]
	    y = ds.gridlon_236[:]
	    xgrid = ds.gridlat_236[:]
	    ygrid = ds.gridlon_236[:]


	    # [OUTPUT] Grid on destination points grid (or read the 1D lat and lon from
	    #	       an other .nc file.
	    newlat1D_points=np.linspace(lat2D_curv.min(), lat2D_curv.max(), 100)
	    newlon1D_points=np.linspace(lon2D_curv.min(), lon2D_curv.max(), 100)

	    output = geocat.comp.triple2grid(x, y, data, xgrid, ygrid)
    """

    # todo: Revisit for handling of "meta" argument
    # Basic sanity checks
    if x.shape[0] != y.shape[0] or x.shape[0] != data.shape[data.ndim - 1]:
        raise DimensionError(
            "ERROR triple2grid: The The length of `x` and `y` must be the same as the rightmost dimension of `data` !"
            )
    if x.ndim > 1 or y.ndim > 1:
        raise DimensionError(
            "ERROR triple2grid: `x` and `y` arguments must be one-dimensional array !\n"
            )
    if xgrid.ndim > 1 or ygrid.ndim > 1:
        raise DimensionError(
            "ERROR triple2grid: `xgrid` and `ygrid` arguments must be one-dimensional array !\n"
            )

    # Parsing Options
    options = {}
    if "method" in kwargs:
        if not isinstance(kwargs["method"], int):
            raise TypeError(
                'ERROR triple2grid: `method` arg must be an integer. Set it to either 1 or 0.'
                )
        input_method = np.asarray(kwargs["method"]).astype(np.int_)
        if (input_method != 0) and (input_method != 1):
            raise TypeError(
                'ERROR triple2grid: `method` arg accepts either 0 or 1.')
        options[b'method'] = input_method

        # `distmx` is only applicable when `method`==1
        if input_method:
            if "distmx" in kwargs:
                input_distmx = np.asarray(kwargs["distmx"]).astype(np.float_)
                if input_distmx.size != 1:
                    raise ValueError(
                        "ERROR triple2grid: Provide a scalar value for `distmx` !"
                        )
                options[b'distmx'] = input_distmx

    if "domain" in kwargs:
        input_domain = np.asarray(kwargs["domain"]).astype(np.float_)
        if input_domain.size != 1:
            raise ValueError(
                "ERROR triple2grid: Provide a scalar value for `domain` !")
        options[b'domain'] = input_domain

    msg = kwargs.get("msg", np.nan)
    meta = kwargs.get("meta", False)

    # the input arguments must be convertible to numpy array
    if isinstance(x, xr.DataArray):
        x = x.values
    if isinstance(y, xr.DataArray):
        y = y.values
    if isinstance(data, xr.DataArray):
        data = data.values
    if isinstance(xgrid, xr.DataArray):
        xgrid = xgrid.values
    if isinstance(ygrid, xr.DataArray):
        ygrid = ygrid.values

    if isinstance(data, np.ndarray):
        fo = _ncomp._triple2grid(x, y, data, xgrid, ygrid, options, msg)
    else:
        raise TypeError("triple2grid: the data input argument must be a "
                        "numpy.ndarray or an xarray.DataArray containing a "
                        "numpy.ndarray.")

    if meta and isinstance(input, xr.DataArray):
        raise MetaError(
            "ERROR triple2grid: retention of metadata is not yet supported !")
    else:
        fo = xr.DataArray(fo)

    return fo
