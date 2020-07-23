import dask.array as da
import numpy as np
import xarray as xr
from dask.array.core import map_blocks

from . import _ncomp
from .errors import (ChunkError, DimensionError, MetaError)


def rgrid2rcm(lat1d, lon1d, fi, lat2d, lon2d, msg=None, meta=False):
    """Interpolates data on a rectilinear lat/lon grid to a curvilinear grid like
       those used by the RCM, WRF and NARR models/datasets.

    Args:

        lat1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the latitude coordinates of
	    the regular grid. Must be monotonically increasing.

        lon1d (:class:`numpy.ndarray`):
	    A one-dimensional array that specifies the longitude coordinates of
	    the regular grid. Must be monotonically increasing.

        fi (:class:`numpy.ndarray`):
	    A multi-dimensional array to be interpolated. The rightmost two
	    dimensions (latitude, longitude) are the dimensions to be interpolated.

        lat2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the latitude locations
	    of fi. Because this array is two-dimensional it is not an associated
	    coordinate variable of `fi`.

        lon2d (:class:`numpy.ndarray`):
	    A two-dimensional array that specifies the longitude locations
	    of fi. Because this array is two-dimensional it is not an associated
	    coordinate variable of `fi`.

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
        :class:`numpy.ndarray`: The interpolated grid. A multi-dimensional array of the
	same size as `fi` except that the rightmost dimension sizes have been replaced
	by the sizes of `lat2d` and `lon2d` respectively. Double if `fi` is double,
	otherwise float.

    Description:
        Interpolates data on a rectilinear lat/lon grid to a curvilinear grid, such as those
	used by the RCM (Regional Climate Model), WRF (Weather Research and Forecasting) and
	NARR (North American Regional Reanalysis) models/datasets. No extrapolation is
	performed beyond the range of the input coordinates. The method used is simple inverse
	distance weighting. Missing values are allowed but ignored.

    Examples:

        Example 1: Using rgrid2rcm with :class:`xarray.DataArray` input

        .. code-block:: python

            import numpy as np
            import xarray as xr
            import geocat.comp

            # Open a netCDF data file using xarray default engine and load the data stream
            # input grid and data
            ds_rect = xr.open_dataset("./DATAFILE_RECT.nc")

            # [INPUT] Grid & data info on the source rectilinear
            ht_rect   =ds_rect.SOME_FIELD[:]
            lat1D_rect=ds_rect.gridlat_[:]
            lon1D_rect=ds_rect.gridlon_[:]

            # Open a netCDF data file using xarray default engine and load the data stream
            # for output grid
            ds_curv = xr.open_dataset("./DATAFILE_CURV.nc")

            # [OUTPUT] Grid on destination curvilinear grid (or read the 2D lat and lon from
            #          an other .nc file
            newlat2D_rect=ds_curv.gridlat2D_[:]
            newlon2D_rect=ds_curv.gridlat2D_[:]

            ht_curv = geocat.comp.rgrid2rcm(lat1D_rect, lon1D_rect, ht_rect, newlat2D_curv, newlon2D_curv)


    """

    # todo: Revisit for handling of "meta" argument
    # Basic sanity checks
    if lat2d.shape[0] != lon2d.shape[0] or lat2d.shape[1] != lon2d.shape[1]:
        raise DimensionError(
            "ERROR rgrid2rcm: The output lat2D/lon2D grids must be the same size !"
        )

    if lat2d.shape[0] < 2 or lon2d.shape[0] < 2 or lat2d.shape[
            1] < 2 or lon2d.shape[1] < 2:
        raise DimensionError(
            "ERROR rgrid2rcm: The input/output lat/lon grids must have at least 2 elements !"
        )

    if fi.ndim < 2:
        raise DimensionError(
            "ERROR rgrid2rcm: fi must be at least two dimensions !\n")

    if fi.shape[fi.ndim - 2] != lat1d.shape[0] or fi.shape[fi.ndim -
                                                           1] != lon1d.shape[0]:
        raise DimensionError(
            "ERROR rgrid2rcm: The rightmost dimensions of fi must be (nlat1d x nlon1d),"
            "where nlat1d and nlon1d are the size of the lat1d/lon1d arrays !")

    if isinstance(lat1d, xr.DataArray):
        lat1d = lat1d.values

    if isinstance(lon1d, xr.DataArray):
        lon1d = lon1d.values

    if not isinstance(fi, xr.DataArray):
        fi = xr.DataArray(fi)

    # ensure lat2d and lon2d are numpy.ndarrays
    if isinstance(lat2d, xr.DataArray):
        lat2d = lat2d.values
    if isinstance(lon2d, xr.DataArray):
        lon2d = lon2d.values

    fi_data = fi.data

    if isinstance(fi_data, da.Array):
        chunks = list(fi.chunks)

        # ensure rightmost dimensions of input are not chunked
        if chunks[-2:] != [lon1d.shape, lat1d.shape]:
            raise ChunkError(
                "rgrid2rcm: the two rightmost dimensions of fi must"
                " not be chunked.")

        # ensure rightmost dimensions of output are not chunked
        chunks[-2:] = (lon2d.shape, lat2d.shape)

        fo = map_blocks(_ncomp._rgrid2rcm,
                        lat1d,
                        lon1d,
                        fi_data,
                        lat2d,
                        lon2d,
                        msg,
                        chunks=chunks,
                        dtype=fi.dtype,
                        drop_axis=[fi.ndim - 2, fi.ndim - 1],
                        new_axis=[fi.ndim - 2, fi.ndim - 1])
    elif isinstance(fi_data, np.ndarray):
        fo = _ncomp._rgrid2rcm(lat1d, lon1d, fi_data, lat2d, lon2d, msg)
    else:
        raise TypeError("rgrid2rcm: the fi input argument must be a "
                        "numpy.ndarray, a dask.array.Array, or an "
                        "xarray.DataArray containing either a numpy.ndarray or"
                        " a dask.array.Array.")

    if meta and isinstance(input, xr.DataArray):
        raise MetaError(
            "ERROR rgrid2rcm: retention of metadata is not yet supported !")
    else:
        fo = xr.DataArray(fo)

    return fo
