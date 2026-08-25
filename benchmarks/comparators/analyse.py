"""Tables and scaling exponents over the comparison CSV.

Nothing here measures anything. The CSV is the result; these are scripts
over it, so a table can be rebuilt without a rerun and a reader can
rebuild a different one.

    python benchmarks/comparators/analyse.py target/comparators/comparison.csv

Cross-method efficiency is reported on common quantities only: f(x) at
held-out points and the log predictive density are the only things every
posterior method in the comparison has. sigma is not among them, because
what it means differs between a method with one residual variance and a
method with a variance ensemble, and neither has a structure count the
other can be compared against. Fit metrics sit beside the efficiency
columns rather than being folded into them, so a method that mixes badly
on data it fits well is visible as exactly that.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "suite"))

from compare import ratio_interval  # noqa: E402
from scorecard import posterior  # noqa: E402

#: The quantities a cross-method efficiency number is computed over.
COMMON = ("f",)


def diagnostics(directory: Path, cell: str) -> dict[str, float]:
    """Return the R-hat and effective sample sizes of one cell's draws.

    Computed by the same pinned ArviZ the suite uses, over every method's
    draws in the comparison. A package's own diagnostics are never the
    source of a published number.
    """
    import arviz as az

    path = directory / cell / "draws.csv"
    if not path.exists():
        return {}
    tree = posterior(pd.read_csv(path))
    keep = [name for name in tree.posterior.data_vars if name.startswith(COMMON)]
    if not keep:
        return {}
    rhat = az.rhat(tree, method="rank", var_names=keep)
    bulk = az.ess(tree, method="bulk", var_names=keep)
    tail = az.ess(tree, method="tail", var_names=keep)
    def flat(d):
        return [float(np.ravel(v.values)[0]) for v in d.dataset.data_vars.values()]

    return {
        "rhat_max": max(flat(rhat)),
        "ess_bulk_min": min(flat(bulk)),
        "ess_tail_min": min(flat(tail)),
    }


def enrich(table: pd.DataFrame, directory: Path) -> pd.DataFrame:
    """Add the diagnostics and the efficiency columns to the CSV."""
    rows = [diagnostics(directory, cell) for cell in table["cell"]]
    table = pd.concat([table.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    sweeps = table["draws"] * table["chains"]
    table["ess_bulk_per_second"] = table["ess_bulk_min"] / table["post_warmup_seconds"]
    table["ess_tail_per_second"] = table["ess_tail_min"] / table["post_warmup_seconds"]
    table["ess_bulk_per_sweep"] = table["ess_bulk_min"] / sweeps
    table["seconds_per_sweep"] = table["post_warmup_seconds"] / sweeps
    # The practitioner's number: how long to a usable posterior, warm-up
    # included, because a user pays for warm-up too.
    table["seconds_to_ess_400"] = table["warmup_seconds"] + (
        table["post_warmup_seconds"] * 400.0 / table["ess_bulk_min"]
    )
    return table


def standard_error(values: pd.Series) -> float:
    """Return the standard error of the mean, zero for a single value."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard error over seeds, per method and dataset."""
    columns = [
        "ess_bulk_per_second",
        "ess_tail_per_second",
        "ess_bulk_per_sweep",
        "seconds_per_sweep",
        "seconds_to_ess_400",
        "rhat_max",
        "rmse",
        "lpd",
        "coverage_95",
        "width_95",
        "fit_seconds",
        "peak_rss_bytes",
    ]
    present = [c for c in columns if c in table]
    keys = ["dataset", "n", "p"] + (["m"] if "m" in table else []) + ["method"]
    grouped = table.groupby(keys, as_index=False)
    out = grouped.agg(
        **{c: (c, "mean") for c in present},
        seeds=("seed", "size"),
    )
    errors = grouped.agg(
        **{f"{c}_se": (c, standard_error) for c in present}
    )
    return out.merge(errors, on=keys)


#: The base point the scaling sweeps hold the other variables at. A sweep
#: in one variable reads only the rows where the other two sit at their
#: base values, so an exponent is never fitted across a confounded mix.
SCALING_BASE = {"n": 500, "p": 10, "m": 200}


def exponents(table: pd.DataFrame, column: str, variable: str = "n") -> pd.DataFrame:
    """Fit log(value) against log(variable) per method; report the slope.

    A point ratio at one size says nothing about how a method behaves at
    another. The exponent with its interval is the claim a scaling sweep
    can support; a ratio is not.
    """
    others = [v for v in SCALING_BASE if v != variable and v in table]
    rows = []
    for (dataset, method), group in table.groupby(["dataset", "method"]):
        sizes = group[group[variable] > 0]
        for other in others:
            sizes = sizes[sizes[other] == SCALING_BASE[other]]
        if sizes[variable].nunique() < 3 or column not in sizes:
            continue
        x = np.log(sizes[variable].to_numpy(dtype=float))
        y = np.log(sizes[column].to_numpy(dtype=float))
        fit = stats.linregress(x, y)
        half = stats.t.ppf(0.975, len(x) - 2) * fit.stderr
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "column": column,
                "variable": variable,
                "exponent": fit.slope,
                "ci_low": fit.slope - half,
                "ci_high": fit.slope + half,
                "points": len(x),
            }
        )
    return pd.DataFrame(rows)


def ratios(
    table: pd.DataFrame, column: str, baseline: str = "thiessen"
) -> pd.DataFrame:
    """Every method against the baseline, as a ratio with an interval.

    Kalibera and Jones (2013): a point ratio without an interval is not a
    result. The interval is on the log ratio with Welch-Satterthwaite
    degrees of freedom, from the per-seed means and standard errors.
    """
    keys = ["dataset", "n", "p"] + (["m"] if "m" in table else [])
    rows = []
    for key, group in table.groupby(keys + ["method"]):
        method = key[-1]
        if method == baseline:
            continue
        base = table[table["method"] == baseline]
        for name, value in zip(keys, key):
            base = base[base[name] == value]
        a, b = base[column].dropna(), group[column].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        ratio, low, high = ratio_interval(
            a.mean(), a.std(ddof=1) / np.sqrt(len(a)), len(a),
            b.mean(), b.std(ddof=1) / np.sqrt(len(b)), len(b),
        )
        rows.append(
            {
                **dict(zip(keys, key)),
                "column": column,
                "method": method,
                f"over_{baseline}": ratio,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    directory = args.csv.parent
    table = enrich(pd.read_csv(args.csv), directory)
    if args.out:
        table.to_csv(args.out, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print("Per method and cell, averaged over seeds\n")
        print(summarise(table).to_string(index=False))
        for column in ("seconds_to_ess_400", "ess_bulk_per_second", "fit_seconds"):
            paired = ratios(table, column)
            if not paired.empty:
                print(f"\nRatios over thiessen, 95 per cent intervals: {column}\n")
                print(paired.to_string(index=False))
        for variable in ("n", "p", "m"):
            for column in ("seconds_per_sweep", "fit_seconds"):
                fitted = exponents(table, column, variable)
                if not fitted.empty:
                    print(f"\nScaling in {variable}: {column}\n")
                    print(fitted.to_string(index=False))


if __name__ == "__main__":
    main()
