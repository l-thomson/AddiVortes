"""Resolution of ``random_state`` to the single seed the core takes."""

from __future__ import annotations

from typing import Union

import numpy as np

__all__ = ["SeedLike", "_resolve_seed"]

SeedLike = Union[int, "np.random.Generator", "np.random.RandomState", None]

_U64 = 1 << 64


def _resolve_seed(random_state: SeedLike) -> int:
    """Resolve ``random_state`` to a seed in [0, 2 ** 64).

    An integer passes through unchanged, so the same integer reproduces the
    same draws. Multi-chain fits pass the resolved seed to the core, which
    derives the per-chain seeds from it.

    Parameters
    ----------
    random_state : int, numpy.random.Generator, numpy.random.RandomState or None
        `None` draws fresh entropy from `numpy.random.SeedSequence`; an
        integer is used directly; a generator supplies one draw.

    Returns
    -------
    int
        The seed.

    Raises
    ------
    ValueError
        If an integer lies outside [0, 2 ** 64).
    TypeError
        For any other type.

    Notes
    -----
    `sklearn.utils.check_random_state` is not used: it maps an integer to a
    `RandomState`, which would stop an integer passing through.
    """
    if random_state is None:
        return int(np.random.SeedSequence().entropy) % _U64  # type: ignore[arg-type]
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(0, _U64, dtype=np.uint64))
    if isinstance(random_state, np.random.RandomState):
        high = int(random_state.randint(0, 1 << 32, dtype=np.uint32))
        low = int(random_state.randint(0, 1 << 32, dtype=np.uint32))
        return (high << 32) | low
    if isinstance(random_state, (int, np.integer)):
        seed = int(random_state)
        if not 0 <= seed < _U64:
            raise ValueError(f"random_state must lie in [0, 2 ** 64), got {seed}")
        return seed
    raise TypeError(
        "random_state must be an int, a numpy.random.Generator, a "
        f"numpy.random.RandomState or None, got {type(random_state).__name__}"
    )
