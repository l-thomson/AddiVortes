"""The cost of crossing into the extension, measured against itself.

`Sampler.step(n)` runs `n` sweeps behind one crossing and `step()` called
`n` times runs the same sweeps behind `n` crossings, so the difference
between the two is the per-call boundary cost with the sampling held
identical. No reference implementation is needed and no number is compared
against a stored one.

Run with

    pytest tests/benchmarks --benchmark-columns=median,iqr,ops

and paste the table into the pull request beside the same table from the
revision being replaced. Nothing here asserts: a wall-clock assertion in
CI is a gate on a measurement shared runners cannot make.
"""

from __future__ import annotations

from typing import Any

import pytest
from thiessen import Model, TermParams
from thiessen.sampler import Sampler

from .conftest import SWEEPS, Design

#: Enough sweeps to fit, few enough to keep a benchmark round short.
BURN_IN = 20
DRAWS = 50

#: The shipped default is 200; the benchmarks use it so they track the
#: configuration a user gets.
ENSEMBLE = TermParams(tessellations=200)


def test_fit(benchmark: Any, design: Design) -> None:
    x, y = design
    model = Model(mean_params=ENSEMBLE, burn_in=BURN_IN, draws=DRAWS)
    benchmark(model.fit, x, y, random_state=1)


def test_predict(benchmark: Any, design: Design) -> None:
    x, y = design
    model = Model(mean_params=ENSEMBLE, burn_in=BURN_IN, draws=DRAWS)
    fitted = model.fit(x, y, random_state=1)
    benchmark(fitted.predict, x)


@pytest.mark.parametrize("per_call", [False, True], ids=["batched", "per_sweep"])
def test_step(benchmark: Any, design: Design, per_call: bool) -> None:
    x, y = design
    sampler = Sampler(x, y, mean_params=ENSEMBLE, random_state=1)
    if per_call:

        def loop() -> None:
            for _ in range(SWEEPS):
                sampler.step()

        benchmark(loop)
    else:
        benchmark(sampler.step, SWEEPS)
