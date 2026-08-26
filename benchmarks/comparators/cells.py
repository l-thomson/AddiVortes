"""The comparison grid: which method runs on which data at which size.

A cell is one (method, dataset, size, seed). The grid is small and fixed,
and every method sees every cell, so a table row is never missing because
a method was quietly dropped from a case it does badly on.

Only stable models appear. Nothing behind the `experimental` Cargo feature
reaches the CSV or the tables: a published comparison of an option that
may change in a patch release is a claim about nothing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _size(name: str, default: int) -> int:
    """A schedule value, overridable for a smoke run.

    The defaults are the published schedule. An override is recorded in
    every cell's metadata and in the CSV, so a table taken at a shortened
    schedule cannot be mistaken for a published one.
    """
    return int(os.environ.get(f"COMPARATOR_{name}", default))


#: The methods, in the order the tables list them. `thiessen` is this
#: library through its Python package; `addivortes` is upstream AddiVortes
#: from GitHub at the commit the lockfile pins; `dbarts` and `bart` are the two BART
#: implementations in common use; `stochtree` is the newer one;
#: `xgboost` is the accuracy baseline and has no posterior, so it carries
#: fit metrics alone.
METHODS = ("thiessen", "addivortes", "dbarts", "bart", "stochtree", "xgboost")

#: The methods whose adapter is an R script rather than a Python module.
R_METHODS = ("addivortes", "dbarts", "bart")

#: Sizes of the main comparison. Small enough that six methods over three
#: processes and three seeds finish in an evening on one machine.
SIZES = ((200, 10), (1000, 10), (1000, 40))

#: The generated processes. Friedman alone favours tree models, so the
#: oblique boundary and the correlated design are here to say where the
#: ranking changes.
PROCESSES = ("friedman", "oblique", "correlated")

#: The real datasets, run at their own size.
REAL = ("diabetes", "attitude")

#: Seeds per cell. Three is the floor; `run.py` adds more where the
#: observed variance asks for them.
SEEDS = (1, 2, 3)

#: The sweep schedule every MCMC method runs. Held identical across
#: methods: a comparison at different schedules is a comparison of
#: schedules.
BURN_IN = _size("BURN_IN", 500)
DRAWS = _size("DRAWS", 4000)
CHAINS = _size("CHAINS", 4)

#: Ensemble size, the value Stone and Gosling (2025) and the BART
#: literature both use as a default.
ENSEMBLE = _size("ENSEMBLE", 200)

#: Threads per cell. One on the main and scaling grids, where the tables
#: compare work per core; `run.py` sets it per cell on the cores grid.
#: Every adapter passes it to its method explicitly, never through
#: `OMP_NUM_THREADS` alone, because not every method reads that.
THREADS = _size("THREADS", 1)

#: The grid a cell is on, set by `run.py`; `thiessen_py` fits the chains
#: as one pooled fit on the cores grid at every core count, so the
#: scaling there reads along one code path.
GRID = os.environ.get("COMPARATOR_GRID", "main")

#: The core counts of the cores grid: one is the grid's own baseline at
#: the headline sizes, and four is the chain count, beyond which no
#: method here has anything to run in parallel.
CORES = (1, 2, 4)

#: Held-out rows, as a share of the dataset.
HOLDOUT = 0.25

#: Held-out rows whose posterior f(x) draws are declared quantities. The
#: only quantities every posterior method has in common are f(x) at
#: held-out points and the log predictive density, so those are the only
#: ones a cross-method efficiency number is computed over.
DECLARED_ROWS = 5

#: The scaling sweeps, one variable at a time on Friedman #1, everything
#: else held at the base point. An exponent fitted over a sweep is the
#: claim; a point ratio at one size is not.
SCALING_BASE = {"n": 500, "p": 10, "m": ENSEMBLE}
SCALING_N = (200, 500, 1000, 2000)
SCALING_P = (5, 10, 20, 40)
SCALING_M = (50, 100, 200, 400)

#: The methods the scaling sweeps run: every MCMC method. XGBoost is an
#: accuracy baseline, not a sampler, and has no sweep cost to fit an
#: exponent to.
SCALING_METHODS = ("thiessen", "addivortes", "dbarts", "bart", "stochtree")


@dataclass(frozen=True)
class Cell:
    """One comparison cell.

    `m` of zero means the default ensemble size; `cores` is the thread
    count the method runs its chains on, one everywhere but the cores
    grid.
    """

    method: str
    dataset: str
    n: int
    p: int
    seed: int
    m: int = 0
    cores: int = 1

    @property
    def id(self) -> str:
        suffix = f"-m{self.m}" if self.m else ""
        if self.cores != 1:
            suffix += f"-c{self.cores}"
        return f"{self.method}-{self.dataset}-n{self.n}-p{self.p}{suffix}-s{self.seed}"

    @property
    def data_id(self) -> str:
        """The cell's data, which every method and core count share."""
        return f"{self.dataset}-n{self.n}-p{self.p}-s{self.seed}"

    @property
    def ensemble(self) -> int:
        return self.m or ENSEMBLE


def grid(methods: tuple[str, ...] = METHODS) -> list[Cell]:
    """Every cell of the main comparison."""
    cells = []
    for method in methods:
        for process in PROCESSES:
            for n, p in SIZES:
                for seed in SEEDS:
                    cells.append(Cell(method, process, n, p, seed))
        for name in REAL:
            for seed in SEEDS:
                cells.append(Cell(method, name, 0, 0, seed))
    return cells


def scaling_grid(methods: tuple[str, ...] = SCALING_METHODS) -> list[Cell]:
    """Every cell of the scaling sweeps, the shared base point once."""
    points = {(n, SCALING_BASE["p"], SCALING_BASE["m"]) for n in SCALING_N}
    points |= {(SCALING_BASE["n"], p, SCALING_BASE["m"]) for p in SCALING_P}
    points |= {(SCALING_BASE["n"], SCALING_BASE["p"], m) for m in SCALING_M}
    cells = []
    for method in methods:
        for n, p, m in sorted(points):
            for seed in SEEDS:
                cells.append(Cell(method, "friedman", n, p, seed, m))
    return cells


def cores_grid(methods: tuple[str, ...] = SCALING_METHODS) -> list[Cell]:
    """Every cell of the cores grid: the main sizes at each core count.

    Wall-clock with the chains on `cores` threads, beside the one-core
    table and never joined to it. XGBoost is an accuracy baseline with no
    chains to spread, so it is not on this grid.
    """
    cells = []
    for method in methods:
        for process in PROCESSES:
            for n, p in SIZES:
                for cores in CORES:
                    for seed in SEEDS:
                        cells.append(Cell(method, process, n, p, seed, cores=cores))
    return cells
