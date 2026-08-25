"""The data-generating processes and real datasets of the comparison.

Friedman #1 alone favours tree models: its mean is additive in four of its
five active covariates and the fifth enters as a product of two, so
axis-aligned splits fit it well. A comparison that runs only Friedman
measures the benchmark, not the methods. Two more processes are here for
that reason, one with an oblique boundary and one with correlated
covariates, and the real datasets carry whatever structure they carry.

Every process writes a plain CSV: covariate columns, then `y`. A method
adapter reads that file and nothing else, so no two adapters can be given
different data by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Dataset:
    """A design, a response, and the name the tables use."""

    name: str
    x: Array
    y: Array

    def write(self, path: Path) -> None:
        header = ",".join(f"x{i + 1}" for i in range(self.x.shape[1])) + ",y"
        table = np.column_stack([self.x, self.y])
        np.savetxt(path, table, delimiter=",", header=header, comments="", fmt="%.17e")


def friedman(n: int, p: int, seed: int, sigma: float = 1.0) -> Dataset:
    """Friedman (1991) benchmark #1: the field's standard case."""
    if p < 5:
        raise ValueError("Friedman #1 reads five covariates")
    rng = np.random.default_rng(seed)
    x = rng.random((n, p))
    mean = (
        10.0 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20.0 * (x[:, 2] - 0.5) ** 2
        + 10.0 * x[:, 3]
        + 5.0 * x[:, 4]
    )
    return Dataset(f"friedman-n{n}-p{p}", x, mean + sigma * rng.standard_normal(n))


def oblique(n: int, p: int, seed: int, sigma: float = 1.0) -> Dataset:
    """A rotated step and a curved ridge: awkward for axis-aligned splits.

    The mean is a function of two rotated coordinates, so a boundary that
    a tessellation cell can follow in one piece needs a staircase of
    axis-aligned splits to approximate.
    """
    if p < 2:
        raise ValueError("the oblique process reads two covariates")
    rng = np.random.default_rng(seed)
    x = rng.random((n, p))
    u = (x[:, 0] - x[:, 1]) / np.sqrt(2.0)
    v = (x[:, 0] + x[:, 1]) / np.sqrt(2.0)
    mean = 8.0 * np.tanh(6.0 * u) + 6.0 * np.exp(-8.0 * (v - 0.7) ** 2)
    return Dataset(f"oblique-n{n}-p{p}", x, mean + sigma * rng.standard_normal(n))


def correlated(
    n: int, p: int, seed: int, sigma: float = 1.0, rho: float = 0.9
) -> Dataset:
    """Friedman #1 on strongly correlated covariates.

    Correlation is where a method that picks one covariate per split and
    one that measures distance over several of them behave differently,
    and it is the case real designs usually are.
    """
    if p < 5:
        raise ValueError("the correlated process reads five covariates")
    rng = np.random.default_rng(seed)
    cov = rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p)))
    z = rng.multivariate_normal(np.zeros(p), cov, size=n)
    # Back to the unit cube through the normal CDF, so the process is
    # Friedman #1 with the same marginals and a correlated copula.
    from scipy.stats import norm

    x = norm.cdf(z)
    mean = (
        10.0 * np.sin(np.pi * x[:, 0] * x[:, 1])
        + 20.0 * (x[:, 2] - 0.5) ** 2
        + 10.0 * x[:, 3]
        + 5.0 * x[:, 4]
    )
    return Dataset(f"correlated-n{n}-p{p}", x, mean + sigma * rng.standard_normal(n))


#: The base R `attitude` data (Chatterjee and Price 1977), 30 rows: the
#: rating response, then complaints, privileges, learning, raises,
#: critical, advance. Reproduced as values so the Python harness needs no
#: R to build its datasets; the upstream fixtures use the same rows.
ATTITUDE = np.array(
    [
        [43, 51, 30, 39, 61, 92, 45], [63, 64, 51, 54, 63, 73, 47],
        [71, 70, 68, 69, 76, 86, 48], [61, 63, 45, 47, 54, 84, 35],
        [81, 78, 56, 66, 71, 83, 47], [43, 55, 49, 44, 54, 49, 34],
        [58, 67, 42, 56, 66, 68, 35], [71, 75, 50, 55, 70, 66, 41],
        [72, 82, 72, 67, 71, 83, 31], [67, 61, 45, 47, 62, 80, 41],
        [64, 53, 53, 58, 58, 67, 34], [67, 60, 47, 39, 59, 74, 41],
        [69, 62, 57, 42, 55, 63, 25], [68, 83, 83, 45, 59, 77, 35],
        [77, 77, 54, 72, 79, 77, 46], [81, 90, 50, 72, 60, 54, 36],
        [74, 85, 64, 69, 79, 79, 63], [65, 60, 65, 75, 55, 80, 60],
        [65, 70, 46, 57, 75, 85, 46], [50, 58, 68, 54, 64, 78, 52],
        [50, 40, 33, 34, 43, 64, 33], [64, 61, 52, 62, 66, 80, 41],
        [53, 66, 52, 50, 63, 80, 37], [40, 37, 42, 58, 50, 57, 49],
        [63, 54, 42, 48, 66, 75, 33], [66, 77, 66, 63, 88, 76, 72],
        [78, 75, 58, 74, 80, 78, 49], [48, 57, 44, 45, 51, 83, 38],
        [85, 85, 71, 71, 77, 74, 55], [82, 82, 39, 59, 64, 78, 39],
    ],
    dtype=np.float64,
)


def real(name: str) -> Dataset:
    """A standard real dataset, by name.

    `diabetes` ships with scikit-learn and needs no network. `attitude` is
    the base R dataset the upstream fixtures already use, reproduced here
    so both suites score the same rows.
    """
    if name == "diabetes":
        from sklearn.datasets import load_diabetes

        data = load_diabetes()
        return Dataset("diabetes", np.asarray(data.data), np.asarray(data.target))
    if name == "attitude":
        return Dataset("attitude", ATTITUDE[:, 1:], ATTITUDE[:, 0])
    raise SystemExit(f"unknown real dataset {name!r}")


#: The processes by name, for the cell grid.
PROCESSES = {
    "friedman": friedman,
    "oblique": oblique,
    "correlated": correlated,
}
