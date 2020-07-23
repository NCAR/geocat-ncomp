import numpy as np
import xarray as xr

from . import _ncomp


def grid2triple(x, y, z, msg=None, meta=False):
    """Converts a two-dimensional grid with one-dimensional coordinate variables
       to an array where each grid value is associated with its coordinates.

    Args:

	x (:class:`numpy.ndarray`):
            Coordinates associated with the right dimension of the variable `z`.
            It must be the same dimension size (call it mx) as the right
            dimension of `z`.

	y (:class:`numpy.ndarray`):
            Coordinates associated with the left dimension of the variable `z`.
            It must be the same dimension size (call it ny) as the left
            dimension of `z`.

	z (:class:`numpy.ndarray`):
            Two-dimensional array of size ny x mx containing the data values.
            Missing values may be present in `z`, but they are ignored.

	msg (:obj:`numpy.number`):
	    A numpy scalar value that represent a missing value in `z`.
	    This argument allows a user to use a missing value scheme
	    other than NaN or masked arrays, similar to what NCL allows.

	meta (:obj:`bool`):
        If set to True and the input array is an Xarray, the metadata
        from the input array will be copied to the output array;
        default is False.
        Warning: this option is not currently supported.

    Returns:
	:class:`numpy.ndarray`: If any argument is "double" the return type
        will be "double"; otherwise a "float" is returned.

    Description:
        The maximum size of the returned array will be 3 x ld where ld <= ny*mx.
        If no missing values are encountered in z, then ld=ny*mx. If missing
        values are encountered in z, they are not returned and hence ld will be
        equal to ny*mx minus the number of missing values found in z. The return
        array will be double if any of the input arrays are double, and float
        otherwise.

    Examples:

	Example 1: Using grid2triple with :class:`xarray.DataArray` input

	.. code-block:: python

	    import numpy as np
	    import xarray as xr
	    import geocat.comp

	    # Open a netCDF data file using xarray default engine and load the data stream
	    ds = xr.open_dataset("./NETCDF_FILE.nc")

	    # [INPUT] Grid & data info on the source curvilinear
	    z=ds.DIST_236_CBL[:]
	    x=ds.gridlat_236[:]
	    y=ds.gridlon_236[:]

	    output = geocat.comp.grid2triple(x, y, z)
    """

    # todo: Revisit for handling of "meta" argument
    # Basic sanity checks
    if z.ndim != 2:
        raise DimensionError(
            "ERROR grid2triple: `z` must be two dimensions !\n")

    if isinstance(x, xr.DataArray):
        x = x.values
    if isinstance(y, xr.DataArray):
        y = y.values
    if isinstance(z, xr.DataArray):
        z = z.values

    if isinstance(z, np.ndarray):
        fo = _ncomp._grid2triple(x, y, z, msg)
    else:
        raise TypeError("grid2triple: the z input argument must be a "
                        "numpy.ndarray or an xarray.DataArray containing a "
                        "numpy.ndarray.")

    if meta and isinstance(input, xr.DataArray):
        raise MetaError(
            "ERROR grid2triple: retention of metadata is not yet supported !")
    else:
        fo = xr.DataArray(fo)

    return fo
