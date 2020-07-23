import dask.array as da
import numpy as np
import xarray as xr
from dask.array.core import map_blocks


def rcm2rgrid(lat2d, lon2d, fi, lat1d, lon1d, msg=None, meta=False):
    """Interpolates data on a curvilinear grid (i.e. RCM, WRF, NARR) to a rectilinear grid.

    Args:

        lat2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the latitudes locations
	    of fi. Because this array is two-dimensional it is not an associated
	    coordinate variable of `fi`. The latitude order must be south-to-north.

        lon2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the longitude locations
	    of fi. Because this array is two-dimensional it is not an associated
	    coordinate variable of `fi`. The latitude order must be west-to-east.

        fi (:class:`numpy.ndarray`):
	    A multi-dimensional array to be interpolated. The rightmost two
	    dimensions (latitude, longitude) are the dimensions to be interpolated.

        lat1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the latitude coordinates of
	    the regular grid. Must be monotonically increasing.

        lon1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the longitude coordinates of
	    the regular grid. Must be monotonically increasing.

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
	replaced by the sizes of lat1d and lon1d respectively.
	Double if fi is double, otherwise float.

    Description:
        Interpolates RCM (Regional Climate Model), WRF (Weather Research and Forecasting) and
        NARR (North American Regional Reanalysis) grids to a rectilinear grid. Actually, this
	function will interpolate most grids that use curvilinear latitude/longitude grids.
	No extrapolation is performed beyond the range of the input coordinates. Missing values
	are allowed but ignored.

	The weighting method used is simple inverse distance squared. Missing values are allowed
	but ignored.

	The code searches the input curvilinear grid latitudes and longitudes for the four
	grid points that surround a specified output grid coordinate. Because one or more of
	these input points could contain missing values, fewer than four points
	could be used in the interpolation.

	Curvilinear grids which have two-dimensional latitude and longitude coordinate axes present
	some issues because the coordinates are not necessarily monotonically increasing. The simple
	search algorithm used by rcm2rgrid is not capable of handling all cases. The result is that,
	sometimes, there are small gaps in the interpolated grids. Any interior points not
	interpolated in the initial interpolation pass will be filled using linear interpolation.
        In some cases, edge points may not be filled.

    Examples:

        Example 1: Using rcm2rgrid with :class:`xarray.DataArray` input

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

            # [OUTPUT] Grid on destination rectilinear grid (or read the 1D lat and lon from
            #          an other .nc file.
            newlat1D_rect=np.linspace(lat2D_curv.min(), lat2D_curv.max(), 100)
            newlon1D_rect=np.linspace(lon2D_curv.min(), lon2D_curv.max(), 100)

            ht_rect = geocat.comp.rcm2rgrid(lat2D_curv, lon2D_curv, ht_curv, newlat1D_rect, newlon1D_rect)


    """

    # todo: Revisit for handling of "meta" argument
    # Basic sanity checks
    if lat2d.shape[0] != lon2d.shape[0] or lat2d.shape[1] != lon2d.shape[1]:
        raise DimensionError(
            "ERROR rcm2rgrid: The input lat/lon grids must be the same size !")

    if lat2d.shape[0] < 2 or lon2d.shape[0] < 2 or lat2d.shape[
            1] < 2 or lon2d.shape[1] < 2:
        raise DimensionError(
            "ERROR rcm2rgrid: The input/output lat/lon grids must have at least 2 elements !"
        )

    if fi.ndim < 2:
        raise DimensionError(
            "ERROR rcm2rgrid: fi must be at least two dimensions !\n")

    if fi.shape[fi.ndim - 2] != lat2d.shape[0] or fi.shape[fi.ndim -
                                                           1] != lon2d.shape[1]:
        raise DimensionError(
            "ERROR rcm2rgrid: The rightmost dimensions of fi must be (nlat2d x nlon2d),"
            "where nlat2d and nlon2d are the size of the lat2d/lon2d arrays !")

    if isinstance(lat2d, xr.DataArray):
        lat2d = lat2d.values

    if isinstance(lon2d, xr.DataArray):
        lon2d = lon2d.values

    if not isinstance(fi, xr.DataArray):
        fi = xr.DataArray(fi)

    # ensure lat1d and lon1d are numpy.ndarrays
    if isinstance(lat1d, xr.DataArray):
        lat1d = lat1d.values
    if isinstance(lon1d, xr.DataArray):
        lon1d = lon1d.values

    fi_data = fi.data

    if isinstance(fi_data, da.Array):
        chunks = list(fi.chunks)

        # ensure rightmost dimensions of input are not chunked
        if chunks[-2:] != [lon2d.shape, lat2d.shape]:
            raise ChunkError(
                "rcm2rgrid: the two rightmost dimensions of fi must"
                " not be chunked.")

        # ensure rightmost dimensions of output are not chunked
        chunks[-2:] = (lon1d.shape, lat1d.shape)

        fo = map_blocks(_ncomp._rcm2rgrid,
                        lat2d,
                        lon2d,
                        fi_data,
                        lat1d,
                        lon1d,
                        msg,
                        chunks=chunks,
                        dtype=fi.dtype,
                        drop_axis=[fi.ndim - 2, fi.ndim - 1],
                        new_axis=[fi.ndim - 2, fi.ndim - 1])
    elif isinstance(fi_data, np.ndarray):
        fo = _ncomp._rcm2rgrid(lat2d, lon2d, fi_data, lat1d, lon1d, msg)
    else:
        raise TypeError("rcm2rgrid: the fi input argument must be a "
                        "numpy.ndarray, a dask.array.Array, or an "
                        "xarray.DataArray containing either a numpy.ndarray or"
                        " a dask.array.Array.")

    if meta and isinstance(input, xr.DataArray):
        raise MetaError(
            "ERROR rcm2rgrid: retention of metadata is not yet supported !")
    else:
        fo = xr.DataArray(fo)

    return fo
