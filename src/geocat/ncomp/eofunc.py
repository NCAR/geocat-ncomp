from typing import Iterable

import numpy as np
import xarray as xr

from . import _ncomp


def eofunc(data: Iterable, neval, **kwargs) -> xr.DataArray:
    """
    Computes empirical orthogonal functions (EOFs, aka: Principal Component Analysis).

    Args:
        data:
            an iterable object containing numbers. It must be at least a 2-dimensional array. The right-most dimension
            is assumed to be the number of observations. Generally this is the time time dimension. If your right-most
            dimension is not time, you could pass ``time_dim=x`` as an argument to define which dimension must be
            treated as time and/or number of observations. Data must be convertible to numpy.array
        neval:
            A scalar integer that specifies the number of eigenvalues and eigenvectors to be returned. This is usually
            less than or equal to the minimum number of observations or number of variables.
        **kwargs:
            extra options controlling the behavior of the function. Currently the following are supported:
            - ``jopt``: a string that indicates whether to use the covariance matrix or the correlation
                        matrix. The default is to use the covariance matrix.
            - ``pcrit``: a float value between ``0`` and ``100`` that indicates the percentage of non-missing points
                         that must exist at any single point in order to be calculated. The default is 50%. Points that
                         contain all missing values will automatically be set to missing.
            - ''time_dim``: an integer defining the time dimension. it must be between ``0`` and ``data.ndim - 1`` or it
                            could be ``-1`` indicating the last dimension. The default value is -1.
            - ``missing_value``: a value defining the missing value. The default is ``np.nan``.
            - ``meta``:  If set to True and the input array is an Xarray, the metadata from the input array will be
                         copied to the output array; default is False.
    """

    # Parsing Options
    options = {}
    if "jopt" in kwargs:
        if not isinstance(kwargs["jopt"], str):
            raise TypeError(
                'jopt must be a string set to either "correlation" or "covariance".'
                )
        if str.lower(kwargs["jopt"]) not in {"covariance", "correlation"}:
            raise ValueError(
                "jopt must be set to either covariance or correlation.")

        options[b'jopt'] = np.asarray(1) if str.lower(
            kwargs["jopt"]) == "correlation" else np.asarray(0)

    if "pcrit" in kwargs:
        provided_pcrit = np.asarray(kwargs["pcrit"]).astype(np.float64)
        if provided_pcrit.size != 1:
            raise ValueError("Only a single number must be provided for pcrit.")

        if (provided_pcrit >= 0.0) and (provided_pcrit <= 100.0):
            options[b'pcrit'] = provided_pcrit
        else:
            raise ValueError("pcrit must be between 0 and 100")

    missing_value = kwargs.get("missing_value", np.nan)

    # the input data must be convertible to numpy array
    np_data = None
    if isinstance(data, np.ndarray):
        np_data = data
    elif isinstance(data, xr.DataArray):
        np_data = data.data
    else:
        np_data = np.asarray(data)

    time_dim = int(kwargs.get("time_dim", -1))

    if (time_dim >= np_data.ndim) or (time_dim < -np_data.ndim):
        raise ValueError(
            f"dimension out of bound. The input data has {np_data.ndim} dimension."
            f" hence, time_dim must be between {-np_data.ndim} and {np_data.ndim - 1}"
            )

    if time_dim < 0:
        time_dim = np_data.ndim + time_dim

    # checking neval
    accepted_neval = int(neval)
    if accepted_neval <= 0:
        raise ValueError("neval must be a positive non-zero integer value.")

    if (time_dim == (np_data.ndim - 1)):
        response = _ncomp._eofunc(np_data,
                                  accepted_neval,
                                  options,
                                  missing_value=missing_value)
    else:
        response = _ncomp._eofunc_n(np_data,
                                    accepted_neval,
                                    time_dim,
                                    options,
                                    missing_value=missing_value)

    attrs = data.attrs if isinstance(data, xr.DataArray) and bool(
        kwargs.get("meta", False)) else {}
    attrs["_FillValue"] = np.nan
    attrs["missing_value"] = np.nan

    # converting the keys to string instead of bytes also fixing matrix and method
    # TODO: once Kevin's work on char * is merged, we could remove this part or change it properly.
    for k, v in response[1].items():
        if k in {b'matrix', b'method'}:
            attrs[k.decode('utf-8')] = v.tostring().decode('utf-8')[:-1]
        else:
            attrs[k.decode('utf-8')] = v

    if isinstance(data, xr.DataArray) and bool(kwargs.get("meta", False)):
        dims = ["evn"
                ] + [data.dims[i] for i in range(data.ndim) if i != time_dim]
        coords = {
            k: v for (k, v) in data.coords.items() if k != data.dims[time_dim]
            }
    else:
        dims = ["evn"
                ] + [f"dim_{i}" for i in range(np_data.ndim) if i != time_dim]
        coords = {}

    return xr.DataArray(response[0], attrs=attrs, dims=dims, coords=coords)


def eofunc_ts(data: Iterable, evec, **kwargs) -> xr.DataArray:
    """
    Calculates the time series of the amplitudes associated with each eigenvalue in an EOF.
    Args:
        data: An Iterable convertible to `numpy.ndarray` in which the rightmost dimension is the number of
              observations. Generally, this is the time dimension. If your rightmost dimension is not time, then pass
              `time_dim` as an extra options.
        evec: An Iterable convertible to `numpy.ndarray` containing the EOFs calculated using `eofunc`.
        **kwargs:
            extra options controlling the behavior of the function. Currently the following are supported:
            - ``jopt``: a string that indicates whether to use the covariance matrix or the correlation
                        matrix. The default is to use the covariance matrix.
            - ''time_dim``: an integer defining the time dimension. it must be between ``0`` and ``data.ndim - 1`` or it
                            could be ``-1`` indicating the last dimension. The default value is -1.
            - ``missing_value``: defines the missing_value. The default is ``np.nan``.
            - ``meta``: If set to True and the input array is an Xarray, the metadata from the input array will be
                        copied to the output array; default is False.

    Returns: A two-dimensional array dimensioned by the number of eigenvalues selected in `eofunc` by the size of the
             time dimension of data. Will contain the following attribute:
             - `ts_mean`: an array of the same size and type as `evec` containing the means removed from data as part
                          of the calculation.

    Examples:
        * Passing a xarray:

        >>> # Openning a data set:
        ... ds = xr.open_dataset("dataset.nc")
        >>> # Extracting SST (Sea Surface temperature)
        ... sst = ds.sst
        >>> evec = eofunc(sst, 5)
        >>> ts = eofunc(sst, evec)

        * Passing a numpy array:

        >>> # Openning a data set:
        ... ds = xr.open_dataset("dataset.nc")
        >>> # Extracting SST (Sea Surface temperature) as Numpy Array
        ... sst = ds.sst.data
        >>> evec = eofunc(sst, 5)
        >>> ts = eofunc(sst, evec.data)

        * Transferring the attributes from input to the output:

        >>> # Openning a data set:
        ... ds = xr.open_dataset("dataset.nc")
        >>> # Extracting SST (Sea Surface temperature)
        ... sst = ds.sst
        >>> evec = eofunc(sst, 5)
        >>> ts = eofunc(sst, evec, meta=True)

        * Defining the time dimension:

        >>> # Openning a data set:
        ... ds = xr.open_dataset("dataset.nc")
        >>> # Extracting SST (Sea Surface temperature)
        ... sst = ds.sst
        >>> evec = eofunc(sst, 5, time_dim=0)
        >>> ts = eofunc(sst, evec, time_dim=0)


    """
    # Parsing Options
    options = {}
    if "jopt" in kwargs:
        if not isinstance(kwargs["jopt"], str):
            raise TypeError(
                'jopt must be a string set to either "correlation" or "covariance".'
                )
        if str.lower(kwargs["jopt"]) not in {"covariance", "correlation"}:
            raise ValueError(
                "jopt must be set to either covariance or correlation.")

        options[b'jopt'] = np.asarray(1) if str.lower(
            kwargs["jopt"]) == "correlation" else np.asarray(0)

    missing_value = kwargs.get("missing_value", np.nan)

    # the input data must be convertible to numpy array
    if isinstance(data, np.ndarray):
        np_data = data
    elif isinstance(data, xr.DataArray):
        np_data = data.data
    else:
        np_data = np.asarray(data)

    # the input data must be convertible to numpy array
    if isinstance(evec, np.ndarray):
        np_evec = evec
    elif isinstance(evec, xr.DataArray):
        np_evec = evec.data
    else:
        np_evec = np.asarray(evec)

    time_dim = int(kwargs.get("time_dim", -1))

    if (time_dim >= np_data.ndim) or (time_dim < -np_data.ndim):
        raise ValueError(
            f"dimension out of bound. The input data has {np_data.ndim} dimension."
            f" hence, time_dim must be between {-np_data.ndim} and {np_data.ndim - 1}"
            )
    if time_dim < 0:
        time_dim = np_data.ndim + time_dim

    if (time_dim == (np_data.ndim - 1)):
        response = _ncomp._eofunc_ts(np_data,
                                     np_evec,
                                     options,
                                     missing_value=missing_value)
    else:
        response = _ncomp._eofunc_ts_n(np_data,
                                       np_evec,
                                       time_dim,
                                       options,
                                       missing_value=missing_value)

    attrs = data.attrs if isinstance(data, xr.DataArray) and bool(
        kwargs.get("meta", False)) else {}
    attrs["_FillValue"] = np.nan
    attrs["missing_value"] = np.nan

    # converting the keys to string instead of bytes also fixing matrix and method
    # TODO: once Kevin's work on char * is merged, we could remove this part or change it properly.
    for k, v in response[1].items():
        if k in {b'matrix'}:
            attrs[k.decode('utf-8')] = v.tostring().decode('utf-8')[:-1]
        else:
            attrs[k.decode('utf-8')] = v

    dims = ["neval", "time"]
    if isinstance(data, xr.DataArray) and bool(kwargs.get("meta", False)):
        coords = {"time": data.coords[data.dims[time_dim]]}
    else:
        coords = {}

    return xr.DataArray(response[0], attrs=attrs, dims=dims, coords=coords)
