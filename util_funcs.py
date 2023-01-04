import numpy as np
from itertools import islice
from typing import Generator, Any


def get_nth_item(gen: Generator, n: int) -> Any:
    """Return the nth item from a generator"""
    return next(islice(gen, n, None), None)


def get_gen_index(gen: Generator, value: [int]) -> int:
    """Get index of generator value that matches value.

    Used to translate from requirement value back to request integer,
    which is the index of the matching table row.

    Args:
        gen: Generator to retrieve value-index from.
        value: list of values to match to generator
    """
    for n, item in enumerate(gen):
        if (item == np.array(value)).all():
            return n
