"""Isolate legacy global RNGs used by cooperating Studio computations.

Process workers remain parallel. In one process, runs using NumPy/Python's
global RNGs must execute serially so they cannot overwrite each other's seed.
"""
from contextlib import contextmanager
from functools import wraps
from inspect import signature
import random
from threading import RLock

import numpy as np

_RNG_LOCK = RLock()


@contextmanager
def seeded_random_state(seed):
    with _RNG_LOCK:
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            np.random.seed(int(seed))
            random.seed(int(seed))
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)


def isolate_random_state(function):
    """Scope an entry point's explicit/default ``seed`` argument."""
    call_signature = signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        bound = call_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        with seeded_random_state(bound.arguments['seed']):
            return function(*args, **kwargs)
    return wrapped
