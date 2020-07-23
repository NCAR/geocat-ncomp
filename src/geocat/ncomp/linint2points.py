import numpy as np
import xarray as xr

from . import _ncomp


def linint2_points(fi, xo, yo, icycx, msg=None, meta=False, xi=None, yi=None):
    """Interpolates from a rectilinear grid to an unstructured grid or locations using bilinear interpolation.

    Args:

        fi (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            An array of two or more dimensions. The two rightmost
            dimensions (nyi x nxi) are the dimensions to be used in
            the interpolation. If user-defined missing values are
            present (other than NaNs), the value of `msg` must be
            set appropriately.

        xo (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            A One-dimensional array that specifies the X (longitude)
            coordinates of the unstructured grid.

        yo (:class:`xarray.DataArray` or :class:`numpy.ndarray`):
            A One-dimensional array that specifies the Y (latitude)
            coordinates of the unstructured grid. It must be the same
            length as `xo`.

        icycx (:obj:`bool`):
            An option to indicate whether the rightmost dimension of fi
            is cyclic. This should be set to True only if you have
            global data, but your longitude values don't quite wrap all
            the way around the globe. For example, if your longitude
            values go from, say, -179.75 to 179.75, or 0.5 to 359.5,
            then you would set this to True.

        msg (:obj:`numpy.number`):
            A numpy scalar value that represent a missing value in fi.
            This argument allows a user to use a missing value scheme
            other than NaN or masked arrays, similar to what NCL allows.

        meta (:obj:`bool`):
            If set to True and the input array is an Xarray, the metadata
            from the input array will be copied to the output array;
            default is False.
            Warning: this option is not currently supported.

        xi (:class:`numpy.ndarray`):
            A strictly monotonically increasing array that specifies
            the X [longitude] coordinates of the `fi` array.

        yi (:class:`numpy.ndarray`):
            A strictly monotonically increasing array that specifies
            the Y [latitude] coordinates of the `fi` array.

    Returns:
	:class:`numpy.ndarray`: The returned value will have the same
        dimensions as `fi`, except for the rightmost dimension which will
        have the same dimension size as the length of `yo` and `xo`. The
        return type will be double if fi is double, and float otherwise.

    Description:
        The inint2_points uses bilinear interpolation to interpolate from
        a rectilinear grid to an unstructured grid.

        If missing values are present, then linint2_points will perform the
        piecewise linear interpolation at all points possible, but will return
        missing values at coordinates which could not be used. If one or more
        of the four closest grid points to a particular (xo,yo) coordinate
        pair are missing, then the return value for this coordinate pair will
        be missing.

        If the user inadvertently specifies output coordinates (xo,yo) that
        are outside those of the input coordinates (xi,yi), the output value
        at this coordinate pair will be set to missing as no extrapolation
        is performed.

        linint2_points is different from linint2 in that `xo` and `yo` are
        coordinate pairs, and need not be monotonically increasing. It is
        also different in the dimensioning of the return array.

        This function could be used if the user wanted to interpolate gridded
        data to, say, the location of rawinsonde sites or buoy/xbt locations.

        Warning: if xi contains longitudes, then the xo values must be in the
        same range. In addition, if the xi values span 0 to 360, then the xo
        values must also be specified in this range (i.e. -180 to 180 will not work).

    Examples:

        Example 1: Using linint2_points with :class:`xarray.DataArray` input

        .. code-block:: python

            import numpy as np
            import xarray as xr
            import geocat.comp

            fi_np = np.random.rand(30, 80)  # random 30x80 array

            # xi and yi do not have to be equally spaced, but they are
            # in this example
            xi = np.arange(80)
            yi = np.arange(30)

            # create target coordinate arrays, in this case use the same
            # min/max values as xi and yi, but with different spacing
            xo = np.linspace(xi.min(), xi.max(), 100)
            yo = np.linspace(yi.min(), yi.max(), 50)

            # create :class:`xarray.DataArray` and chunk it using the
            # full shape of the original array.
            # note that xi and yi are attached as coordinate arrays
            fi = xr.DataArray(fi_np,
                              dims=['lat', 'lon'],
                              coords={'lat': yi, 'lon': xi}
                             ).chunk(fi_np.shape)

            fo = geocat.comp.linint2_points(fi, xo, yo, 0)

    """

    # todo: Revisit for handling of "meta" argument
    # Basic sanity checks
    if not isinstance(fi, xr.DataArray):
        fi = xr.DataArray(fi)
        if xi is None or yi is None:
            raise CoordinateError(
                "linint2_points: arguments xi and yi must be passed"
                " explicitly if fi is not an xarray.DataArray !")
    if xo.shape[0] != yo.shape[0]:
        raise DimensionError(
            "ERROR linint2_points: The xo and yo must be the same size !")

    if fi.ndim < 2:
        raise DimensionError(
            "ERROR linint2_points: fi must be at least two dimensions !\n")

    if xi is None:
        xi = fi.coords[fi.dims[-1]].values
    elif isinstance(xi, xr.DataArray):
        xi = xi.values

    if yi is None:
        yi = fi.coords[fi.dims[-2]].values
    elif isinstance(yi, xr.DataArray):
        yi = yi.values

    if isinstance(xo, xr.DataArray):
        xo = xo.values
    if isinstance(yo, xr.DataArray):
        yo = yo.values

    fi_data = fi.values

    if isinstance(fi_data, np.ndarray):
        fo = _ncomp._linint2_points(xi, yi, fi_data, xo, yo, icycx, msg)
    else:
        raise TypeError

    if meta and isinstance(input, xr.DataArray):
        raise MetaError(
            "ERROR linint2_points: retention of metadata is not yet supported !"
        )
    else:
        fo = xr.DataArray(fo)

    # todo: Revisit for the parallelization:
    # Above two if-blocks should be changed with two if-blocks similar to the following (would require corrections
    # though) when parallelization for differently-shaped input (fi) and output (fo) arrays in this case is resolved:
    # if isinstance(fi_data, da.Array):
    #     chunks = list(fi.chunks)
    #
    #     # ensure rightmost dimensions of input are not chunked
    #     if chunks[-2:] != [yi.shape, xi.shape]:
    #         raise ChunkError("linint2_points: the two rightmost dimensions of fi must not be chunked.")
    #
    #     # Ensure rightmost dimensions of output are not chunked
    #     # chunks[-2:] = [yo.shape, xo.shape]
    #
    #     # map_blocks maps each chunk of fi_data to a separate invocation of _ncomp._linint2_points. The "chunks"
    #     # keyword argument should be the chunked dimensionality of the expected output; the number of chunks should
    #     # match that of fi_data. Additionally, "drop_axis" and "new_axis" in this case indicate that the two rightmost
    #     # dimensions of the input will be dropped from the output array, and that two new axes will be added instead.
    #     fo = map_blocks(_ncomp._linint2_points, xi, yi, fi_data, xo, yo, icycx, msg,
    #                     chunks=chunks, dtype=fi.dtype,
    #                     drop_axis=[fi.ndim-2, fi.ndim-1],
    #                     new_axis=[fi.ndim-2, fi.ndim-1])
    #
    # elif isinstance(fi_data, np.ndarray):
    #     fo = _ncomp._linint2_points(xi, yi, fi_data, xo, yo, icycx, msg)
    #
    # else:
    #     raise TypeError
    #
    # if meta:
    #     coords = {k:v if k not in fi.dims[-2:]
    #               else (xo if k == fi.dims[-1] else yo)
    #               for (k, v) in fi.coords.items()}
    #
    #     fo = xr.DataArray(fo, attrs=fi.attrs, dims=fi.dims,
    #                           coords=coords)
    # else:
    #     fo = xr.DataArray(fo)
    return fo
