"""Run the comparison and write one tidy CSV.

Harness discipline, each clause of it earned by something that goes wrong
without it:

- One subprocess per cell. Peak resident set becomes attributable, a
  method that leaks does not poison the next one, and a run resumes from
  whatever it has already written.
- Randomised run order under a fixed shuffle seed, with the first cell
  discarded as a warm-up. Machines drift over a long run: thermal
  throttling, another process waking. In a fixed order that drift is
  confounded with the method; shuffled, it becomes noise, and fixing the
  shuffle seed keeps the run reproducible.
- A thread count per cell, set in the environment of every library that
  reads one there and passed to every method explicitly by its adapter.
  One on the main and scaling grids, so the tables compare work per
  core: a comparison where one method uses eight cores and another one
  is a comparison of thread counts. The cores grid runs the same cells
  at one, two and four threads and is reported apart, as wall-clock.
- A user-against-elapsed guard. If a cell's user time exceeds its elapsed
  time times its thread count, by more than the 5 per cent slack that
  process start-up and the runtime's own threads take, the pinning did
  not take and the cell's timing is void.
- A held-out accuracy guard. A method that is fast because it is wrong is
  the failure mode a timing table cannot see; a cell whose held-out error
  is far worse than the best method's on the same data is flagged.

    python benchmarks/comparators/run.py --out target/comparators

Options:

    --grid main|scaling|cores   the main comparison, the scaling sweeps,
                                or the main sizes at one, two and four
                                threads
    --methods NAME [NAME ...]   only these methods
    --datasets NAME [NAME ...]  only these processes or real datasets
    --sizes N,P [N,P ...]       only these sizes
    --seeds N [N ...]           only these seeds, and no variance rounds
    --resume                    skip cells that already have a meta.json
    --csv PATH                  where the tidy CSV goes

Repetitions are sized from observed variance: after the base seeds, any
cell group whose fit-time relative standard error is above the target
gets another seed, round by round, up to a cap.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from cells import (  # noqa: E402
    BURN_IN,
    CHAINS,
    DECLARED_ROWS,
    DRAWS,
    HOLDOUT,
    METHODS,
    R_METHODS,
    SEEDS,
    Cell,
    cores_grid,
    grid,
    scaling_grid,
)
from datasets import PROCESSES, Dataset, real  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

#: The shuffle seed. Fixed, so the order is random and reproducible at
#: once; a run at a different seed is a different run and says so in the
#: metadata.
SHUFFLE_SEED = 20260825

#: The slack the user-against-elapsed guard allows: interpreter start-up,
#: the runtime's own threads and the parent's bookkeeping run beside the
#: fit, and a cell that stays within it ran on its thread count.
THREAD_SLACK = 1.05


def thread_env(cores: int) -> dict[str, str]:
    """The environment that sets every library's thread count to `cores`.

    These are the variables the numerical libraries in this comparison
    read; the methods that take a thread count as an argument (dbarts,
    stochtree, this library) get it from `COMPARATOR_THREADS` through
    their adapter, because none of them reads `OMP_NUM_THREADS`.
    """
    count = str(cores)
    return {
        "OMP_NUM_THREADS": count,
        "OPENBLAS_NUM_THREADS": count,
        "MKL_NUM_THREADS": count,
        "NUMEXPR_NUM_THREADS": count,
        "VECLIB_MAXIMUM_THREADS": count,
        "COMPARATOR_THREADS": count,
    }


#: A cell whose held-out error is worse than this multiple of the best on
#: the same data is flagged: fast because wrong is the failure a timing
#: table cannot see.
ACCURACY_GUARD = 1.5

#: Repetitions per cell are sized from observed variance: seeds are added
#: to a cell group until the relative standard error of its fit time is
#: below the target or the cap is reached.
RSE_TARGET = 0.05
MAX_SEEDS = 6


def dataset_files(cell: Cell, out: Path) -> tuple[Path, Path]:
    """Write the cell's data once and return the train and test paths.

    Every method at a cell reads the same two files, so no two methods can
    be given different data by accident.
    """
    directory = out / "data"
    directory.mkdir(parents=True, exist_ok=True)
    train = directory / f"{cell.data_id}-train.csv"
    test = directory / f"{cell.data_id}-test.csv"
    if train.exists() and test.exists():
        return train, test

    if cell.dataset in PROCESSES:
        data = PROCESSES[cell.dataset](cell.n, cell.p, cell.seed)
    else:
        data = real(cell.dataset)
    rows = data.x.shape[0]
    order = random.Random(cell.seed).sample(range(rows), rows)
    cut = int(round(rows * (1.0 - HOLDOUT)))
    Dataset(data.name, data.x[order[:cut]], data.y[order[:cut]]).write(train)
    Dataset(data.name, data.x[order[cut:]], data.y[order[cut:]]).write(test)
    return train, test


def command(cell: Cell, train: Path, test: Path, out: Path) -> list[str]:
    if cell.method in R_METHODS:
        return [
            "Rscript",
            str(HERE / "adapters" / "r_methods.R"),
            cell.method,
            str(train),
            str(test),
            str(out),
            str(cell.seed),
            str(BURN_IN),
            str(DRAWS),
            str(CHAINS),
            str(cell.ensemble),
            str(DECLARED_ROWS),
            str(cell.cores),
        ]
    module = {"thiessen": "thiessen_py", "xgboost": "xgboost_py"}.get(
        cell.method, f"{cell.method}_py"
    )
    return [
        sys.executable,
        "-m",
        f"adapters.{module}",
        str(train),
        str(test),
        str(out),
        str(cell.seed),
    ]


def run_cell(cell: Cell, out: Path, grid: str) -> dict | None:
    """Run one cell in its own process; return its metadata."""
    directory = out / cell.id
    train, test = dataset_files(cell, out)
    environment = {
        **os.environ,
        **thread_env(cell.cores),
        "PYTHONPATH": str(HERE),
        "COMPARATOR_ENSEMBLE": str(cell.ensemble),
        "COMPARATOR_GRID": grid,
    }
    started = time.perf_counter()
    invocation = command(cell, train, test, directory)
    process = subprocess.Popen(invocation, env=environment, cwd=HERE)
    _, status, usage = os.wait4(process.pid, 0)
    elapsed = time.perf_counter() - started
    if status != 0:
        print(f"{cell.id}: exited with status {status}", file=sys.stderr)
        return None
    meta = json.loads((directory / "meta.json").read_text())
    scale = 1 if sys.platform == "darwin" else 1024
    meta["cores"] = cell.cores
    meta["peak_rss_bytes"] = usage.ru_maxrss * scale
    # `wait4` reports the child and every descendant it reaped, so the
    # forked workers of the R adapters count towards the guard.
    meta["user_seconds"] = usage.ru_utime
    meta["system_seconds"] = usage.ru_stime
    meta["elapsed_seconds"] = elapsed
    # Written back, so a resumed run reads the same row a fresh one would.
    (directory / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def run_cells(cells: list[Cell], args: argparse.Namespace) -> list[dict]:
    """Run the cells in the given order and return their rows."""
    rows = []
    for index, cell in enumerate(cells, start=1):
        done = args.out / cell.id / "meta.json"
        if args.resume and done.exists():
            meta = json.loads(done.read_text())
        else:
            print(f"[{index}/{len(cells)}] {cell.id}", file=sys.stderr)
            meta = run_cell(cell, args.out, args.grid)
        if meta is None:
            continue
        meta.update(
            {
                "cell": cell.id,
                "data": cell.data_id,
                "dataset": cell.dataset,
                "n": cell.n,
                "p": cell.p,
                "m": cell.ensemble,
                "cores": cell.cores,
                "seed": cell.seed,
                "shuffle_seed": SHUFFLE_SEED,
                "schedule": f"{BURN_IN}+{DRAWS}x{CHAINS}@{cell.ensemble}",
            }
        )
        rows.append(meta)
    return rows


def underpowered(rows: list[dict], seen: dict[str, Cell]) -> list[Cell]:
    """The next round of cells for the groups whose fit time is noisy.

    One more seed per group whose relative standard error of fit time is
    above the target, until the seed cap.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (
            row["method"],
            row["dataset"],
            row["n"],
            row["p"],
            row["m"],
            row["cores"],
        )
        groups.setdefault(key, []).append(row)
    extra = []
    for members in groups.values():
        if len(members) >= MAX_SEEDS:
            continue
        times = [r["fit_seconds"] for r in members]
        mean = sum(times) / len(times)
        variance = sum((t - mean) ** 2 for t in times) / (len(times) - 1)
        if mean > 0 and (variance / len(times)) ** 0.5 / mean > RSE_TARGET:
            latest = max(members, key=lambda r: r["seed"])
            done = seen[latest["cell"]]
            extra.append(
                Cell(
                    done.method,
                    done.dataset,
                    done.n,
                    done.p,
                    done.seed + 1,
                    done.m,
                    done.cores,
                )
            )
    return extra


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "target" / "comparators")
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--grid", choices=("main", "scaling", "cores"), default="main")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--sizes", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.grid == "scaling":
        cells = scaling_grid(tuple(m for m in args.methods if m != "xgboost"))
    elif args.grid == "cores":
        cells = cores_grid(tuple(m for m in args.methods if m != "xgboost"))
    else:
        cells = grid(tuple(args.methods))
    if args.datasets:
        cells = [c for c in cells if c.dataset in set(args.datasets)]
    if args.sizes:
        wanted = {tuple(int(v) for v in size.split(",")) for size in args.sizes}
        cells = [c for c in cells if (c.n, c.p) in wanted or c.n == 0]
    if args.seeds:
        cells = [c for c in cells if c.seed in set(args.seeds)]
    if not cells:
        raise SystemExit("no cells selected")

    random.Random(SHUFFLE_SEED).shuffle(cells)
    # The first cell pays for a cold cache and a cold clock; it is run and
    # its numbers are dropped.
    warmup = cells[0]
    print(f"warm-up: {warmup.id}, discarded", file=sys.stderr)
    run_cell(warmup, args.out / "warmup", args.grid)

    args.out.mkdir(parents=True, exist_ok=True)
    seen = {cell.id: cell for cell in cells}
    rows = run_cells(cells, args)

    # Repetitions sized from observed variance: noisy groups get more
    # seeds, one round at a time, each round in its own shuffled order.
    if not args.seeds:
        for round_index in range(1, MAX_SEEDS - len(SEEDS) + 1):
            extra = underpowered(rows, seen)
            if not extra:
                break
            print(f"variance round {round_index}: {len(extra)} cells", file=sys.stderr)
            random.Random(SHUFFLE_SEED + round_index).shuffle(extra)
            seen.update({cell.id: cell for cell in extra})
            rows.extend(run_cells(extra, args))

    table = pd.DataFrame(rows)
    table["threads_ok"] = (
        table["user_seconds"]
        <= table["elapsed_seconds"] * table["cores"] * THREAD_SLACK
    )
    best = table.groupby("data")["rmse"].transform("min")
    table["accuracy_ok"] = table["rmse"] <= best * ACCURACY_GUARD
    destination = args.csv or args.out / "comparison.csv"
    table.to_csv(destination, index=False)
    print(f"wrote {destination}", file=sys.stderr)

    for flag, message in (
        (
            "threads_ok",
            "thread pinning did not take: user time above elapsed times cores",
        ),
        ("accuracy_ok", "held-out error far worse than the best on the same data"),
    ):
        bad = table[~table[flag]]
        for _, row in bad.iterrows():
            print(f"  {row['cell']}: {message}", file=sys.stderr)


if __name__ == "__main__":
    main()
