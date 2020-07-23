import numpy as np
import xarray as xr

from . import _ncomp
# The following imports allow for the function name to be used directly under the package namespace, skipping the module name.
# This is done to maintain backwards compatibily from when the functions were defined in geocat/ncomp/__init__.py
from .errors import (
    AttributeError, DimensionError, MetaError)


def dpres_plevel(plev, psfc, ptop=None, msg=None, meta=False):
    """Calculates the pressure layer thicknesses of a constant pressure level coordinate system.

    Args:

        plev (:class:`numpy.ndarray`):
            A one dimensional array containing the constant pressure levels. May be
            in ascending or descending order. Must have the same units as `psfc`.

        psfc (:class:`numpy.ndarray`):
            A scalar or an array of up to three dimensions containing the surface
            pressure data in Pa or hPa (mb). The rightmost dimensions must be latitude
            and longitude. Must have the same units as `plev`.

        ptop (:class:`numpy.number`):
            A scalar specifying the top of the column. ptop should be <= min(plev).
            Must have the same units as `plev`.

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
        :class:`numpy.ndarray`: If psfc is a scalar the return variable will be a
        one-dimensional array the same size as `plev`; if `psfc` is two-dimensional
        [e.g. (lat,lon)] or three-dimensional [e.g. (time,lat,lon)] then the return
        array will have an additional level dimension: (lev,lat,lon) or (time,lev,lat,lon).
        The returned type will be double if `psfc` is double, float otherwise.

    Description:
        Calculates the layer pressure thickness of a constant pressure level system. It
        is analogous to `dpres_hybrid_ccm` for hybrid coordinates. At each grid point the
        sum of the pressure thicknesses equates to [psfc-ptop]. At each grid point, the
        returned values above `ptop` and below `psfc` will be set to the missing value of `psfc`.
        If there is no missing value for `psfc` then the missing value will be set to the default
        for float or double appropriately. If `ptop` or `psfc` is between plev levels
        then the layer thickness is modifed accordingly. If `psfc` is set to a missing value, all
        layer thicknesses are set to the appropriate missing value.

        The primary purpose of this function is to return layer thicknesses to be used to
        weight observations for integrations.

    Examples:

        Example 1: Using dpres_plevel with :class:`xarray.DataArray` input

        .. code-block:: python

            import numpy as np
            import xarray as xr
            import geocat.comp

            # Open a netCDF data file using xarray default engine and load the data stream
            ds = xr.open_dataset("./SOME_NETCDF_FILE.nc")

            # [INPUT] Grid & data info on the source
            psfc = ds.PS
            plev = ds.LEV
            ptop = 0.0

            # Call the function
            result_dp = geocat.comp.dpres_plevel(plev, psfc, ptop)
    """

    # todo: Revisit for handling of "meta" argument
    if isinstance(psfc, np.ndarray):
        if psfc.ndim > 3:
            raise DimensionError(
                "ERROR dpres_plevel: The 'psfc' array must be a scalar or be a 2 or 3 dimensional array with right most dimensions lat x lon !"
                )
    if plev.ndim != 1:
        raise DimensionError(
            "ERROR dpres_plevel: The 'plev' array must be 1 dimensional array !"
            )
    if isinstance(ptop, np.ndarray):
        raise DimensionError(
            "ERROR dpres_plevel: The 'ptop' value must be a scalar !")
    if isinstance(plev, xr.DataArray) and isinstance(psfc, xr.DataArray):
        if plev.attrs["units"] != psfc.attrs["units"]:
            raise AttributeError(
                "ERROR dpres_plevel: Units of 'plev' and 'psfc' needs to match !"
                )

    if isinstance(plev, xr.DataArray):
        plev = plev.values

    if isinstance(psfc, xr.DataArray):
        psfc = psfc.values
    elif np.size(psfc) == 1:  # if it is a scalar, then construct a ndarray
        psfc = np.asarray(psfc)
        psfc = np.ndarray([1], buffer=psfc, dtype=psfc.dtype)

    if ptop is None:
        ptop = min(plev)
    else:
        if ptop > min(plev):
            raise ValueError(
                "ERROR dpres_plevel: The 'ptop' value must be <= min(plev) !")

    # call the ncomp 'dpres_plevel' function
    result_dp = _ncomp._dpres_plevel(plev, psfc, ptop, msg)

    if meta and isinstance(input, xr.DataArray):
        raise MetaError(
            "ERROR dpres_plevel: retention of metadata is not yet supported !")

        pass  # TODO: Retaining possible metadata might be revised in the future
    else:
        result_dp = xr.DataArray(result_dp)

    return result_dp
