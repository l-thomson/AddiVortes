"""The design the binding benchmarks run on.

Friedman (1991) benchmark #1, generated here rather than drawn from numpy's
global state, so a benchmark measures the same work on every run and on
every machine.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

#: Rows and columns of the small cell. A copy of the design shows as a
#: slope across the two sizes rather than as an offset, so both are needed.
SMALL = (200, 10)

#: Rows and columns of the large cell.
LARGE = (2000, 10)

#: Sweeps in the per-call cases. Large enough that the boundary cost of one
#: crossing per sweep is visible beside the sweep itself.
SWEEPS = 200

SEED = 20260824

Design = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


def friedman(n: int, p: int) -> Design:
    """Return the Friedman #1 design and response at `n` rows, `p` columns."""
    rng = np.random.default_rng(SEED)
    x = rng.random((n, p))
    y = (
        10.0 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20.0 * (x[:, 2] - 0.5) ** 2
        + 10.0 * x[:, 3]
        + 5.0 * x[:, 4]
        + rng.standard_normal(n)
    )
    return x, y


@pytest.fixture(params=[SMALL, LARGE], ids=["n200", "n2000"])
def design(request: pytest.FixtureRequest) -> Design:
    """The Friedman #1 design at each benchmark size."""
    n, p = request.param
    return friedman(n, p)
