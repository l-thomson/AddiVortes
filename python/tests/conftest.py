"""Fixtures shared across the test suite.

`gaussian_fixture` is the core's fixed-seed fixture
(`crates/thiessen/tests/common/mod.rs`), reproduced so that the seed test can
compare against the core's stored snapshot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from thiessen import TermParams

SEED = 7


def sweep(tessellations: int, burn_in: int, draws: int) -> dict[str, Any]:
    """Return a sweep schedule as constructor arguments."""
    return {
        "mean_params": TermParams(tessellations=tessellations),
        "burn_in": burn_in,
        "draws": draws,
    }


#: Short enough to keep the suite quick, long enough to fit.
SMALL = sweep(8, 10, 20)


def survival(events: Any, times: Any) -> npt.NDArray[Any]:
    """Return a structured survival array in the scikit-survival layout."""
    return np.array(list(zip(events, times)), dtype=[("event", bool), ("time", float)])


Fixture = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


def _fixture() -> Fixture:
    n = 48
    x = np.array(
        [[i / (n - 1), ((i * 37) % n) / n] for i in range(n)],
        dtype=np.float64,
    )
    # The multiplications associate as the core's fixture does; `3.0 * d ** 2`
    # rounds differently from `(3.0 * d) * d` and moves the chain.
    y = np.array(
        [
            3.0 * (x[i, 0] - 0.4) * (x[i, 0] - 0.4)
            + 0.5 * x[i, 1]
            + 0.3 * (((i * 29) % 17) / 16.0 - 0.5)
            for i in range(n)
        ],
        dtype=np.float64,
    )
    return x, y


@pytest.fixture
def gaussian_fixture() -> Fixture:
    """The core's fixed-seed fixture: 48 rows, two covariates."""
    return _fixture()


@pytest.fixture
def probit_fixture() -> Fixture:
    """The Gaussian fixture with the response thresholded at its median."""
    x, y = _fixture()
    return x, (y > np.median(y)).astype(np.float64)
