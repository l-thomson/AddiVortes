"""The package's draws against the core's stored chain.

Bit-exact on the reference target `x86_64-unknown-linux-gnu`, which is where
the core's snapshot is checked; the test is skipped elsewhere, as the core's
own snapshot test compares posterior summaries there.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pytest
from thiessen import Model

#: The seed of the core's fixed-seed fixture.
SEED = 7

SNAPSHOT = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "thiessen"
    / "tests"
    / "chains"
    / "gaussian.txt"
)

#: The fixture rows the core snapshots f(x) at.
POINTS = [0, 17, 33]

reference_target = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() != "x86_64",
    reason="the snapshot is bit-exact on x86_64-unknown-linux-gnu only",
)


def _stored() -> np.ndarray:
    """Parse the snapshot: one row per draw, sigma then f(x) at each point."""
    lines = SNAPSHOT.read_text().splitlines()
    return np.array(
        [[float(field) for field in line.split()] for line in lines[1:] if line],
        dtype=np.float64,
    )


@reference_target
@pytest.mark.skipif(not SNAPSHOT.is_file(), reason="snapshot not in the source tree")
def test_draws_equal_the_core_snapshot(gaussian_fixture):
    x, y = gaussian_fixture
    stored = _stored()

    fitted = Model(m=15, burn_in=50, draws=60).fit(x, y, random_state=SEED)

    assert fitted.n_draws == stored.shape[0]
    np.testing.assert_array_equal(fitted.sigma(), stored[:, 0])
    np.testing.assert_array_equal(fitted.predict_draws(x[POINTS]), stored[:, 1:])
