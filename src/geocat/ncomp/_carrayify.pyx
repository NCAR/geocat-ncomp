# cython: language_level=3, boundscheck=False, embedsignature=True
import functools
import numpy as np


def carrayify(f):
    """
    A decorator that ensures that :class:`numpy.ndarray` arguments are
    C-contiguous in memory. The decorator function takes no arguments.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        new_args = list(args)
        for i, arg in enumerate(new_args):
            if isinstance(arg, np.ndarray) and not arg.flags.carray:
                new_args[i] = np.ascontiguousarray(arg)
        return f(*new_args, **kwargs)
    return wrapper
