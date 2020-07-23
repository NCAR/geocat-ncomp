import numpy as np
import xarray as xr

from . import _ncomp


def moc_globe_atl(lat_aux_grid,
                  a_wvel,
                  a_bolus,
                  a_submeso,
                  tlat,
                  rmlak,
                  msg=None,
                  meta=False):
    """Facilitates calculating the meridional overturning circulation for the
    globe and Atlantic.

    Args:

        lat_aux_grid (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Latitude grid for transport diagnostics.

        a_wvel (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Area weighted Eulerian-mean vertical velocity [TAREA*WVEL].

        a_bolus (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Area weighted Eddy-induced (bolus) vertical velocity [TAREA*WISOP].

        a_submeso (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Area weighted submeso vertical velocity [TAREA*WSUBM].

        tlat (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Array of t-grid latitudes.

        rmlak (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            Basin index number: [0]=Globe, [1]=Atlantic

        msg (:obj:`numpy.number`):
          A numpy scalar value that represent a missing value.
          This argument allows a user to use a missing value scheme
          other than NaN or masked arrays, similar to what NCL allows.

        meta (:obj:`bool`):
            If set to True and the input array is an Xarray, the metadata
            from the input array will be copied to the output array;
            default is False.
            Warning: this option is not currently supported.

        Returns:
            :class:`xarray.DataArray`: A multi-dimensional array of size [moc_comp] x
            [n_transport_reg] x [kdepth] x [nyaux] where:

            - moc_comp refers to the three components returned
            - n_transport_reg refers to the Globe and Atlantic
            - kdepth is the the number of vertical levels of the work arrays
            - nyaux is the size of the lat_aux_grid

            The type of the output data will be double only if a_wvel or a_bolus or
            a_submesa is of type double. Otherwise, the return type will be float.


    Examples:

        # Input data can be read from a data set as follows:
        import xarray as xr

        ds = xr.open_dataset("input.nc")

        lat_aux_grid = ds.lat_aux_grid
        a_wvel = ds.a_wvel
        a_bolus = ds.a_bolus
        a_submeso = ds.a_submeso
        tlat = ds.tlat
        rmlak = ds.rmlak

        # (1) Calling with xArray inputs and default arguments (Missing value = np.nan, NO meta information)
        out_arr = moc_globe_atl(lat_aux_grid, a_wvel, a_bolus, a_submeso, tlat, rmlak)

        # (2) Calling with Numpy inputs and default arguments (Missing value = np.nan, NO meta information)
        out_arr = moc_globe_atl(lat_aux_grid.values, a_wvel.values, a_bolus.values, a_submeso.values,
                                tlat.values, rmlak.values)

        # (3) Calling with xArray inputs and user-defined arguments (Missing value = np.nan, NO meta information)
        out_arr = moc_globe_atl(lat_aux_grid, a_wvel, a_bolus, a_submeso, tlat, rmlak, msg=-99.0, meta=True)

        # (4) Calling with Numpy inputs and user-defined arguments (Missing value = np.nan, NO meta information)
        out_arr = moc_globe_atl(lat_aux_grid.values, a_wvel.values, a_bolus.values, a_submeso.values,
                                tlat.values, rmlak.values, msg=-99.0, meta=True)

    """

    # todo: Revisit for handling of "meta" argument
    # Ensure input arrays are numpy.ndarrays
    if isinstance(lat_aux_grid, xr.DataArray):
        lat_aux_grid = lat_aux_grid.values

    if isinstance(a_wvel, xr.DataArray):
        a_wvel = a_wvel.values

    if isinstance(a_bolus, xr.DataArray):
        a_bolus = a_bolus.values

    if isinstance(a_submeso, xr.DataArray):
        a_submeso = a_submeso.values

    if isinstance(tlat, xr.DataArray):
        tlat = tlat.values

    if isinstance(rmlak, xr.DataArray):
        rmlak = rmlak.values

    # Make sure msg has the correct dtype even if given wrong type or a scalar instead of np.num
    if a_wvel.dtype == np.float64:
        msg = np.float64(msg)
    else:
        msg = np.float32(msg)

    # Call ncomp function
    out_arr = _ncomp._moc_globe_atl(lat_aux_grid, a_wvel, a_bolus, a_submeso,
                                    tlat, rmlak, msg)

    if meta and isinstance(input, xr.DataArray):
        raise MetaError(
            "ERROR moc_globe_atl: retention of metadata is not yet supported !")
    else:
        out_arr = xr.DataArray(out_arr)

    return out_arr
