Wrapping NComp functions using Cython
=====================================
1. Add function signature from NComp to `$GEOCATNCOMP/geocat/ncomp/libncomp.pxd` under the 'cdef extern from "ncomp/wrapper.h"' section. *Make sure to include "nogil" at the end of the function signature to release the Python global interpreter lock, which is essential for multithreaded performace with Dask.*

2. Create a new function in `$GEOCATNCOMP/geocat/ncomp/_ncomp.pyx`, prepended with an underscore (`_linint2` for example). The Cython function signature acts as the "numpy interface" to NComp. All arguments should be explicitly typed as either np.ndarray or an appropriate C type (int, double, etc).

3. Create a unified _ncomp.Array object (which abstracts a NumPy array and an ncomp_array* into a single Python object) for each input np.ndarray using the `_ncomp.Array.from_np` builder method. This `Array` object allocates and deallocates ncomp_array structs as needed.

4. Create output np.ndarray(s) using np.zeros (essentially equivalent to `calloc`ing), again creating an _ncomp.Array object using `_ncomp.Array.from_np`.

5. Call C function from the "libncomp" namespace, "libncomp.linint2" for example, capturing return value (standardized return codes and error handling still to be determined). Ensure function call is inside "`with nogil:`" block. Use the "ncomp" attribute on the `_ncomp.Array.ncomp` arrays as arguments to the C functions from libncomp; the `.ncomp` attribute provides an `ncomp_array*` as expected by the libncomp functions.

6. Return the previously created output np.ndarray; if metadata needs to be returned as well, then instead return a tuple with a dictionary containing metadata keys and values as the second element of the tuple.

Wrapping Cython functions in Python
===================================
1. Create a new function in `$GEOCATNCOMP/geocat/ncomp/__init__.py` (`linint2` for example).

2. Include `meta=True` as a keyword argument for any function that could potentially retain metadata; retaining metadata will be the default behavior.

3. Check for .chunks attribute on the primary input xarray.DataArray, set chunk sizes to be equal to the shape of the array if not already chunked.

4. Call Dask's `map_blocks` function. The first argument is the actual function name from Cython (`_ncomp._linint2` in this case), followed by positional parameters to the desired Cython function, then followed by keyword arguments for `map_blocks`. The `chunks` keyword argument is particularly important as the total number of chunks on the input array and output array must match in order for Dask to properly align data in the output array.

5. Re-attach metadata as needed.

6. Return output.
