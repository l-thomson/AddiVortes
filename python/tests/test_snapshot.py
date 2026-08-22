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
from thiessen import Model, TermParams

#: The seed of the core's fixed-seed fixture.
SEED = 7

CHAINS = (
    Path(__file__).resolve().parents[2] / "crates" / "thiessen" / "tests" / "chains"
)
SNAPSHOT = CHAINS / "gaussian.txt"

#: The fixture rows the core snapshots f(x) at.
POINTS = [0, 17, 33]

reference_target = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine() != "x86_64",
    reason="the snapshot is bit-exact on x86_64-unknown-linux-gnu only",
)


def _stored(path: Path = SNAPSHOT) -> np.ndarray:
    """Parse a chain file: one row per draw, its header naming the columns."""
    lines = path.read_text().splitlines()
    return np.array(
        [[float(field) for field in line.split()] for line in lines[1:] if line],
        dtype=np.float64,
    )


def _core_model(**kwargs):
    return Model(
        mean_params=TermParams(tessellations=15), burn_in=50, draws=60, **kwargs
    )


@reference_target
@pytest.mark.skipif(not SNAPSHOT.is_file(), reason="snapshot not in the source tree")
def test_draws_equal_the_core_snapshot(gaussian_fixture):
    x, y = gaussian_fixture
    stored = _stored()

    fitted = _core_model().fit(x, y, random_state=SEED)

    assert fitted.n_draws == stored.shape[0]
    np.testing.assert_array_equal(fitted.sigma(), stored[:, 0])
    np.testing.assert_array_equal(fitted.predict_draws(x[POINTS]), stored[:, 1:])


@reference_target
@pytest.mark.skipif(not SNAPSHOT.is_file(), reason="snapshot not in the source tree")
def test_probit_draws_equal_the_core_snapshot(gaussian_fixture):
    from thiessen import probit

    x, y = gaussian_fixture
    stored = _stored(CHAINS / "probit.txt")
    # The core's fixture thresholds at the upper middle order statistic.
    threshold = np.sort(y)[y.size // 2]
    labels = (y >= threshold).astype(np.float64)

    fitted = _core_model(outcome=probit()).fit(x, labels, random_state=SEED)

    assert fitted.n_draws == stored.shape[0]
    np.testing.assert_array_equal(fitted.predict_latent(x[POINTS]), stored)


@reference_target
@pytest.mark.skipif(not SNAPSHOT.is_file(), reason="snapshot not in the source tree")
def test_heteroscedastic_draws_equal_the_core_snapshot(gaussian_fixture):
    x, y = gaussian_fixture
    stored = _stored(CHAINS / "heteroscedastic.txt")
    i = np.arange(y.size)
    noise = 0.3 * (((i * 29) % 17) / 16.0 - 0.5)
    scaled = y - noise + noise * (0.2 + 2.0 * x[:, 0])

    model = _core_model(variance_params=TermParams(tessellations=5))
    fitted = model.fit(x, scaled, random_state=SEED)

    assert fitted.n_draws == stored.shape[0]
    np.testing.assert_array_equal(fitted.predict_draws(x[POINTS]), stored[:, :3])
    np.testing.assert_array_equal(fitted.predict_variance(x[POINTS]), stored[:, 3:])
