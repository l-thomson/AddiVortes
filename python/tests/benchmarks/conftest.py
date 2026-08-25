"""The designs the binding benchmarks run on.

Written by the core rather than generated here, so the two sides run on
the same numbers: the core's generator is a splitmix64 that no other
language reproduces, and a comparison over different data is a comparison
of different work. Write them, and the core's own times, with

    cargo run --release --manifest-path bench/Cargo.toml --bin overhead -- \\
        designs target/bindings
    cargo run --release --manifest-path bench/Cargo.toml --bin overhead -- \\
        run > target/bindings/core.json

and point `THIESSEN_BENCH_DESIGNS` elsewhere if that is not where they
went.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

Core = dict[str, Any]

ROOT = Path(__file__).resolve().parents[3]
DESIGNS = Path(os.environ.get("THIESSEN_BENCH_DESIGNS", ROOT / "target" / "bindings"))

Design = tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]


def _core() -> Core:
    path = DESIGNS / "core.json"
    if not path.exists():
        pytest.skip(f"no core timings at {path}; see this module's docstring")
    return dict(json.loads(path.read_text()))


@pytest.fixture(scope="session")
def core() -> Core:
    """The core's own times on the same cases, and the case parameters."""
    return _core()


def _sizes() -> list[tuple[int, int]]:
    if not (DESIGNS / "core.json").exists():
        return [(200, 10)]
    cases = _core_unchecked()["cases"]
    return sorted({(case["n"], case["p"]) for case in cases})


def _core_unchecked() -> Core:
    return dict(json.loads((DESIGNS / "core.json").read_text()))


@pytest.fixture(params=_sizes(), ids=lambda s: f"n{s[0]}-p{s[1]}")
def size(request: pytest.FixtureRequest) -> tuple[int, int]:
    """One (rows, columns) pair of the binding-overhead cases."""
    n, p = request.param
    return int(n), int(p)


@pytest.fixture
def design(size: tuple[int, int]) -> Design:
    """The training design and response the core wrote for this size."""
    n, p = size
    path = DESIGNS / f"train-n{n}-p{p}.csv"
    if not path.exists():
        pytest.skip(f"no design at {path}; see this module's docstring")
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    return table[:, :p], table[:, p]


@pytest.fixture
def predict_design(size: tuple[int, int]) -> npt.NDArray[np.float64]:
    """The predict matrix the core wrote for this column count."""
    _, p = size
    path = DESIGNS / f"predict-p{p}.csv"
    if not path.exists():
        pytest.skip(f"no predict matrix at {path}; see this module's docstring")
    return np.loadtxt(path, delimiter=",", skiprows=1)
