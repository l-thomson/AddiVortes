"""Run the benchmark suite and write its scorecard.

One process per chain, so the peak resident set is attributable and a run
resumes. The cell set comes from the core's registry through `suite list`,
so a model added there is benchmarked without an edit here.

Repetitions are sized from the variance observed so far: the run keeps
adding repetitions of a cell until the standard error of its minimum bulk
ESS per second falls below `--target-rse`, or `--reps-max` is reached.

    python benchmarks/suite/run.py --out target/suite

Options:

    --cells ID [ID ...]   only these cells
    --sizes small|all     the small cell of each model, or every cell
    --chains N            chains per repetition (default 4)
    --reps-min N          repetitions before the variance is consulted
    --reps-max N          repetitions after which the run stops regardless
    --target-rse F        relative standard error the repetitions aim at
    --experimental        include the models behind the Cargo feature
    --scorecard PATH      where the tidy CSV goes
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from scorecard import Run, relative_standard_error, scorecard, summarise  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

#: Base seed of repetition 0. Repetition r uses BASE_SEED + r * STRIDE, and
#: chain c within it takes the core's own chain_seed of that.
BASE_SEED = 20260824
STRIDE = 1_000_003


MANIFEST = ROOT / "bench" / "Cargo.toml"


def build(experimental: bool) -> Path:
    """Build the suite binary in release and return its path.

    The bench crate is outside the root workspace and has a target
    directory of its own, so the path is read from cargo rather than
    assumed.
    """
    command = [
        "cargo",
        "build",
        "--locked",
        "--release",
        "--manifest-path",
        str(MANIFEST),
        "--bin",
        "suite",
    ]
    if experimental:
        command += ["--features", "experimental"]
    subprocess.run(command, check=True)
    metadata = json.loads(
        subprocess.run(
            [
                "cargo",
                "metadata",
                "--locked",
                "--no-deps",
                "--format-version",
                "1",
                "--manifest-path",
                str(MANIFEST),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return Path(metadata["target_directory"]) / "release" / "suite"


def list_cells(binary: Path) -> dict:
    out = subprocess.run(
        [str(binary), "list"], check=True, capture_output=True, text=True
    )
    return json.loads(out.stdout)


def run_chain(binary: Path, cell: dict, seed: int, chain: int, out: Path) -> int:
    """Run one chain in its own process; return its peak resident set."""
    command = [
        str(binary),
        "run",
        cell["model"],
        str(cell["n"]),
        str(cell["p"]),
        str(seed),
        str(chain),
        str(out),
    ]
    process = subprocess.Popen(command)
    _, status, usage = os.wait4(process.pid, 0)
    if status != 0:
        raise SystemExit(f"{cell['id']} chain {chain} failed with status {status}")
    # ru_maxrss is in kilobytes on Linux and in bytes on macOS.
    scale = 1 if sys.platform == "darwin" else 1024
    return usage.ru_maxrss * scale


def repetition(binary: Path, cell: dict, rep: int, chains: int, out: Path) -> Run:
    seed = BASE_SEED + rep * STRIDE
    directory = out / cell["id"] / f"rep{rep}"
    directory.mkdir(parents=True, exist_ok=True)
    peak = 0
    frames = []
    metadata = []
    for chain in range(chains):
        peak = max(peak, run_chain(binary, cell, seed, chain, directory))
        stem = f"{cell['id']}-chain{chain}"
        frames.append(pd.read_csv(directory / f"{stem}.csv"))
        metadata.append(json.loads((directory / f"{stem}.json").read_text()))
    return Run(
        cell=cell["id"],
        model=cell["model"],
        n=cell["n"],
        p=cell["p"],
        seed=seed,
        draws=pd.concat(frames, ignore_index=True),
        metadata=metadata,
        peak_rss_bytes=peak,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "target" / "suite")
    parser.add_argument("--scorecard", type=Path)
    parser.add_argument("--cells", nargs="+")
    parser.add_argument("--sizes", choices=["small", "all"], default="all")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--reps-min", type=int, default=3)
    parser.add_argument("--reps-max", type=int, default=8)
    parser.add_argument("--target-rse", type=float, default=0.05)
    parser.add_argument("--experimental", action="store_true")
    args = parser.parse_args()

    binary = build(args.experimental)
    listing = list_cells(binary)
    cells = listing["cells"]
    if args.cells:
        cells = [c for c in cells if c["id"] in set(args.cells)]
    elif args.sizes == "small":
        smallest = min(c["n"] for c in cells)
        cells = [c for c in cells if c["n"] == smallest]
    if not cells:
        raise SystemExit("no cells selected")

    args.out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for cell in cells:
        cards: list[pd.DataFrame] = []
        while len(cards) < args.reps_max:
            cards.append(
                scorecard(repetition(binary, cell, len(cards), args.chains, args.out))
            )
            if len(cards) < args.reps_min:
                continue
            rse = relative_standard_error(cards, "ess_bulk_min_per_second")
            print(
                f"{cell['id']}: {len(cards)} repetitions, "
                f"relative standard error {rse:.3f}",
                file=sys.stderr,
            )
            if rse <= args.target_rse:
                break
        summaries.append(summarise(cards))

    table = pd.concat(summaries, ignore_index=True)
    table["chains"] = args.chains
    table["draws"] = listing["draws"]
    table["burn_in"] = listing["burn_in"]
    table["core_version"] = json.loads(
        (
            args.out / cells[0]["id"] / "rep0" / f"{cells[0]['id']}-chain0.json"
        ).read_text()
    )["core_version"]
    destination = args.scorecard or args.out / "scorecard.csv"
    table.to_csv(destination, index=False)
    print(f"wrote {destination}", file=sys.stderr)


if __name__ == "__main__":
    main()
