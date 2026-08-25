"""The Python binding's cases, timed against the core's on the same work.

The exposure that matters is per call, not per fit. The sampler API is
callable from Python, so a user's loop crosses the boundary once per sweep
instead of once per fit, and a tax invisible on `fit` is the dominant cost
there. `Sampler.step(n)` runs n sweeps behind one crossing and `step()`
called n times runs them behind n crossings, so the pair measures the
boundary cost with the sampling held identical.

There is no progress case here: the Python binding has no progress
surface, so nothing crosses back once per sweep. The R binding does, and
`benchmarks/bindings/overhead.R` measures it.

Run on one machine, old revision against new, and put the table in the
pull request:

    pip install -r python/requirements-bench.txt
    pytest tests/benchmarks --benchmark-json=../target/bindings/python.json
    python ../benchmarks/bindings/overhead.py ../target/bindings

Nothing here asserts. A wall-clock assertion in CI is a gate on a
measurement shared runners cannot make.
"""

from __future__ import annotations

from typing import Any

import numpy.typing as npt
import pytest
from thiessen import Model, TermParams
from thiessen.sampler import Sampler

from .conftest import Core, Design

#: The schedule and the ensemble of the core's registry workload; the
#: comparison is void if these drift apart.
BURN_IN = 20
DRAWS = 50
ENSEMBLE = TermParams(tessellations=200)


def _model() -> Model:
    return Model(mean_params=ENSEMBLE, burn_in=BURN_IN, draws=DRAWS)


def test_fit(benchmark: Any, design: Design) -> None:
    x, y = design
    benchmark(_model().fit, x, y, random_state=1)


def test_predict(
    benchmark: Any, design: Design, predict_design: npt.NDArray[Any]
) -> None:
    x, y = design
    fitted = _model().fit(x, y, random_state=1)
    benchmark(fitted.predict, predict_design)


@pytest.mark.parametrize("per_call", [False, True], ids=["batched", "per_call"])
def test_sweeps(benchmark: Any, design: Design, core: Core, per_call: bool) -> None:
    x, y = design
    sweeps = int(core["sweeps"])

    def batched() -> None:
        Sampler(x, y, mean_params=ENSEMBLE, random_state=1).step(sweeps)

    def one_at_a_time() -> None:
        sampler = Sampler(x, y, mean_params=ENSEMBLE, random_state=1)
        for _ in range(sweeps):
            sampler.step()

    benchmark(one_at_a_time if per_call else batched)
