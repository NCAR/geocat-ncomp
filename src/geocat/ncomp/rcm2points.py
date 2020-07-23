import numpy as np
import xarray as xr

from . import _ncomp


def rcm2points(lat2d,
               lon2d,
               fi,
               lat1dPoints,
               lon1dPoints,
               opt=0,
               msg=None,
               meta=False):
    """Interpolates data on a curvilinear grid (i.e. RCM, WRF, NARR) to an unstructured grid.

    Args:

	lat2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the latitudes locations
	    of fi. The latitude order must be south-to-north.

	lon2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the longitude locations
	    of fi. The latitude order must be west-to-east.

	fi (:class:`numpy.ndarray`):
	    A multi-dimensional array to be interpolated. The rightmost two
	    dimensions (latitude, longitude) are the dimensions to be interpolated.

	lat1dPoints (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the latitude coordinates of
	    the output locations.

	lon1dPoints (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the longitude coordinates of
	    the output locations.

	opt (:obj:`numpy.number`):
	    opt=0 or 1 means use an inverse distance weight interpolation.
	    opt=2 means use a bilinear interpolation.

	msg (:obj:`numpy.number`):
	    A numpy scalar value that represent a missing value in fi.
	    This argument allows a user to use a missing value scheme
	    other than NaN or masked arrays, similar to what NCL allows.

	meta (:obj:`bool`):
        If set to True and the input array is an Xarray, the metadata
        from the input array will be copied to the output array;
        default is False.
        Warning: this option is not currently supported.

    Returns:
	:class:`numpy.ndarray`: The interpolated grid. A multi-dimensional array
	of the same size as fi except that the rightmost dimension sizes have been
	replaced by the number of coordinate pairs (lat1dPoints, lon1dPoints).
	Double if fi is double, otherwise float.

    Description:
	Interpolates data on a curvilinear grid, such as those used by the RCM (Regional Climate Model),
	WRF (Weather Research and Forecasting) and NARR (North American Regional Reanalysis)
	models/datasets to an unstructured grid. All of these have latitudes that are oriented south-to-north.

	A inverse distance squared algorithm is used to perform the interpolation.

	Missing values are allowed and no extrapolation is performed.

    Examples:

	Example 1: Using rcm2points with :class:`xarray.DataArray` input

	.. code-block:: python

	    import numpy as np
	    import xarray as xr
	    import geocat.comp

	    # Open a netCDF data file using xarray default engine and load the data stream
	    ds = xr.open_dataset("./ruc.nc")

	    # [INPUT] Grid & data info on the source curvilinear
	    ht_curv=ds.DIST_236_CBL[:]
	    lat2D_curv=ds.gridlat_236[:]
	    lon2D_curv=ds.gridlon_236[:]

	    # [OUTPUT] Grid on destination points grid (or read the 1D lat and lon from
	    #	       an other .nc file.
	    newlat1D_points=np.linspace(lat2D_curv.min(), lat2D_curv.max(), 100)
	    newlon1D_points=np.linspace(lon2D_curv.min(), lon2D_curv.max(), 100)

	    ht_points = geocat.comp.rcm2points(lat2D_curv, lon2D_curv, ht_curv, newlat1D_points, newlon1D_points)
    """

    # todo: Revisit for handling of "meta" argument
    # Basic sanity checks
    if lat2d.shape[0] != lon2d.shape[0] or lat2d.shape[1] != lon2d.shape[1]:
        raise DimensionError(
            "ERROR rcm2points: The input lat/lon grids must be the same size !")

    if lat1dPoints.shape[0] != lon1dPoints.shape[0]:
        raise DimensionError(
            "ERROR rcm2points: The output lat/lon grids must be same size !")

    if lat2d.shape[0] < 2 or lon2d.shape[0] < 2 or lat2d.shape[
            1] < 2 or lon2d.shape[1] < 2:
        raise DimensionError(
            "ERROR rcm2points: The input/output lat/lon grids must have at least 2 elements !"
        )

    if fi.ndim < 2:
        raise DimensionError(
            "ERROR rcm2points: fi must be at least two dimensions !\n")

    if fi.shape[fi.ndim - 2] != lat2d.shape[0] or fi.shape[fi.ndim -
                                                           1] != lon2d.shape[1]:
        raise DimensionError(
            "ERROR rcm2points: The rightmost dimensions of fi must be (nlat2d x nlon2d),"
            "where nlat2d and nlon2d are the size of the lat2d/lon2d arrays !")

    if isinstance(lat2d, xr.DataArray):
        lat2d = lat2d.values
    if isinstance(lon2d, xr.DataArray):
        lon2d = lon2d.values
    if not isinstance(fi, xr.DataArray):
        fi = xr.DataArray(fi)

    # ensure lat1d and lon1d are numpy.ndarrays
    if isinstance(lat1dPoints, xr.DataArray):
        lat1dPoints = lat1dPoints.values
    if isinstance(lon1dPoints, xr.DataArray):
        lon1dPoints = lon1dPoints.values

    fi_data = fi.values

    if isinstance(fi_data, np.ndarray):
        fo = _ncomp._rcm2points(lat2d, lon2d, fi_data, lat1dPoints, lon1dPoints,
                                opt, msg)
    else:
        raise TypeError("rcm2points: the fi input argument must be a "
                        "numpy.ndarray, a dask.array.Array, or an "
                        "xarray.DataArray containing either a numpy.ndarray or"
                        " a dask.array.Array.")

    if meta and isinstance(input, xr.DataArray):
        raise MetaError(
            "ERROR rcm2points: retention of metadata is not yet supported !")
    else:
        fo = xr.DataArray(fo)

    return fo
