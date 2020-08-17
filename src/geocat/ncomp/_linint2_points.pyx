import numpy as np
from . cimport libncomp
from . cimport _ncomp
from ._carrayify import carrayify


@carrayify
def _linint2_points(np.ndarray xi_np, np.ndarray yi_np, np.ndarray fi_np, np.ndarray xo_np, np.ndarray yo_np, int icycx, msg=None):
    """_linint2_points(xi, yi, fi, xo, yo, icycx, msg=None)

    Interpolates from a rectilinear grid to an unstructured grid
    or locations using bilinear interpolation.

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
            Set to True for metadata; default is False.

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

    """

    xi = Array.from_np(xi_np)
    yi = Array.from_np(yi_np)
    fi = Array.from_np(fi_np)
    xo = Array.from_np(xo_np)
    yo = Array.from_np(yo_np)

    cdef long i
    if fi.type == libncomp.NCOMP_DOUBLE:
        fo_dtype = np.float64
    else:
        fo_dtype = np.float32

    cdef np.ndarray fo_np = np.zeros(tuple([fi.shape[i] for i in range(fi.ndim - 2)] + [yo.shape[0]]), dtype=fo_dtype)

    replace_fi_nans = False

    if msg is None or np.isnan(msg):  # if no missing value specified, assume NaNs
        missing_inds_fi = np.isnan(fi.numpy)
        msg = get_default_fill(fi.numpy)
        replace_fi_nans = True
    else:
        missing_inds_fi = (fi.numpy == msg)

    set_ncomp_msg(&(fi.ncomp.msg), msg)  # always set missing on fi.ncomp

    if missing_inds_fi.any():
        fi.ncomp.has_missing = 1
        fi.numpy[missing_inds_fi] = msg

    fo = Array.from_np(fo_np)

    cdef int ier
    with nogil:
        ier = libncomp.linint2points(xi.ncomp, yi.ncomp, fi.ncomp,
                                     xo.ncomp, yo.ncomp, fo.ncomp,
                                     icycx)
    if ier:
        warnings.warn("linint2_points: {}: xi, yi, xo, and yo must be monotonically increasing".format(ier),
                      NcompWarning)

    if replace_fi_nans and fi.ncomp.has_missing:
        fi.numpy[missing_inds_fi] = np.nan

    if fo.type == libncomp.NCOMP_DOUBLE:
        fo_msg = fo.ncomp.msg.msg_double
    else:
        fo_msg = fo.ncomp.msg.msg_float

    fo.numpy[fo.numpy == fo_msg] = np.nan

    return fo.numpy
