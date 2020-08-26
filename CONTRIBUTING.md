Please first refer to [GeoCAT Contributor's Guide](https://geocat.ucar.edu/pages/contributing.html) for overall 
contribution guidelines (such as detailed description of GeoCAT structure, forking, repository cloning, 
branching, etc.). Once you determine that a function should be contributed under this repo, please refer to the 
following contribution guidelines:


# Adding new functions to GeoCAT-ncomp repo

## Wrapping libncomp functions using Cython

1. Add function signature from [libncomp](https://github.com/NCAR/libncomp) 
to `$GEOCATNCOMP/src/geocat/ncomp/libncomp.pxd` under the 'cdef extern from "ncomp/wrapper.h"' section. 
*Make sure to include "nogil" at the end of the function signature to release the Python global interpreter lock, 
which is essential for multithreaded performace with Dask.*

2. Create a new function in `$GEOCATNCOMP/src/geocat/ncomp/_ncomp.pyx`, prepended with an underscore 
(`_linint2` for example). The Cython function signature acts as the "numpy interface" to NComp. All arguments should 
be explicitly typed as either np.ndarray or an appropriate C type (int, double, etc).

3. Create a unified _ncomp.Array object (which abstracts a NumPy array and an ncomp_array* into a single Python object) 
for each input np.ndarray using the `_ncomp.Array.from_np` builder method. This `Array` object allocates and deallocates 
ncomp_array structs as needed.

4. Create output np.ndarray(s) using np.zeros (essentially equivalent to `calloc`ing), again creating an _ncomp.Array 
object using `_ncomp.Array.from_np`.

5. Call C function from the "libncomp" namespace, "libncomp.linint2" for example, capturing return value (standardized 
return codes and error handling still to be determined). Ensure function call is inside "`with nogil:`" block. 
Use the "ncomp" attribute on the `_ncomp.Array.ncomp` arrays as arguments to the C functions from libncomp; the `.ncomp` 
attribute provides an `ncomp_array*` as expected by the libncomp functions.

6. Return the previously created output np.ndarray; if metadata needs to be returned as well, then instead return a 
tuple with a dictionary containing metadata keys and values as the second element of the tuple.


## Wrapping Cython functions in Python

1. Create a new function in `$GEOCATNCOMP/src/geocat/ncomp/__init__.py` (`linint2` for example).

2. Include `meta=True` as a keyword argument for any function that could potentially retain metadata; retaining metadata 
will be the default behavior.

3. Check for .chunks attribute on the primary input xarray.DataArray, set chunk sizes to be equal to the shape of the 
array if not already chunked.

4. Call Dask's `map_blocks` function. The first argument is the actual function name from Cython (`_ncomp._linint2` in 
this case), followed by positional parameters to the desired Cython function, then followed by keyword arguments 
for `map_blocks`. The `chunks` keyword argument is particularly important as the total number of chunks on the input 
array and output array must match in order for Dask to properly align data in the output array.

5. Re-attach metadata as needed.

6. Return output.


# Adding unit tests

All new computational functionality needs to include unit testing. For that purpose, please refer to the following 
guideline:

1. Unit tests of the function should be implemented as a separate test file under the `$GEOCATNCOMP/test` folder.

2. The [pytest](https://docs.pytest.org/en/stable/contents.html) testing framework is used as a “runner” for the tests. 
For further information about `pytest`, see: [pytest documentation](https://docs.pytest.org/en/stable/contents.html).
    - Test scripts themselves are not intended to use `pytest` through implementation. Instead, `pytest` should be used 
    only for running test scripts as follows:
    
        `pytest <test_script_name>.py` 

    - Not using `pytest` for implementation allows the unit tests to be also run by using: 

        `python -m unittest <test_script_name>.py`
        
3. Python’s unit testing framework [unittest](https://docs.python.org/3/library/unittest.html) is used for 
implementation of the test scripts. For further information about `unittest`, 
see: [unittest documentation](https://docs.python.org/3/library/unittest.html).

4. Recommended but not mandatory implementation approach is as follows:
    - Common data structures as well as variables and functions, which could be used by multiple test methods throughout 
    the test script, are defined under a base test class.
    - Any group of testing functions dedicated to testing a particular phenomenon (e.g. a specific edge case, data 
    structure, etc.) is implemented by a class, which inherits TestCase from Python’s `unittest` and likely the base 
    test class implemented for the purpose mentioned above.
    - Assertions are used for testing various cases such as array comparison.
    - Please see previously implemented test cases for reference of the recommended testing approach, 
    e.g. [test_moc_globe_atl.py](https://github.com/NCAR/geocat-ncomp/blob/master/test/test_moc_globe_atl.py)
