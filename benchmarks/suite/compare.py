"""Compare two scorecards and report effect sizes with intervals.

A point ratio is not a result. Each metric is reported as the ratio of the
two means with a confidence interval built from the repetition standard
errors each scorecard carries, following Kalibera and Jones (2013): the
interval, not the point, is what says whether anything happened.

    python benchmarks/suite/compare.py <baseline.csv> <new.csv>

Ratios are new over baseline, so a ratio above one means the new
scorecard's value is larger. Whether larger is better depends on the
metric and is left to the reader; the metric names say which is which.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["cell", "model", "n", "p", "metric", "unit"]


def ratio_interval(
    mean_a: float, se_a: float, reps_a: int, mean_b: float, se_b: float, reps_b: int
) -> tuple[float, float, float]:
    """Return the ratio b / a and its 95 per cent confidence interval.

    The interval is built on the log ratio, whose standard error is the
    delta-method combination of the two relative standard errors, with
    Welch-Satterthwaite degrees of freedom. On the log scale the interval
    is symmetric and cannot reach a negative ratio.
    """
    if mean_a == 0.0 or mean_b == 0.0:
        return float("nan"), float("nan"), float("nan")
    ratio = mean_b / mean_a
    rse_a = se_a / abs(mean_a)
    rse_b = se_b / abs(mean_b)
    se_log = math.sqrt(rse_a**2 + rse_b**2)
    if se_log == 0.0:
        return ratio, ratio, ratio
    numerator = (rse_a**2 + rse_b**2) ** 2
    denominator = 0.0
    for rse, reps in ((rse_a, reps_a), (rse_b, reps_b)):
        if reps > 1:
            denominator += rse**4 / (reps - 1)
    df = numerator / denominator if denominator > 0 else 1.0
    half = stats.t.ppf(0.975, df) * se_log
    return ratio, ratio * math.exp(-half), ratio * math.exp(half)


def compare(baseline: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Return one row per metric present in both scorecards."""
    merged = baseline.merge(new, on=KEYS, suffixes=("_base", "_new"))
    rows = []
    for _, row in merged.iterrows():
        ratio, low, high = ratio_interval(
            row["value_base"],
            row["se_base"],
            int(row["reps_base"]),
            row["value_new"],
            row["se_new"],
            int(row["reps_new"]),
        )
        rows.append(
            {
                **{key: row[key] for key in KEYS},
                "baseline": row["value_base"],
                "new": row["value_new"],
                "ratio": ratio,
                "ci_low": low,
                "ci_high": high,
                # An interval straddling one is a measurement that did not
                # separate the two revisions, whatever the point ratio says.
                "separated": bool(
                    np.isfinite(low) and np.isfinite(high) and not (low <= 1.0 <= high)
                ),
            }
        )
    return pd.DataFrame(rows)


#: Metrics a gate may read, and the direction that is a regression. These
#: are ratios per sweep and per held-out row: they do not change with the
#: speed of the machine, so a shared runner can be asked about them. No
#: wall-clock metric appears here or ever will.
#:
#: R-hat is absent: mixing is what a regression moves, and effective
#: sample size per sweep measures it on a scale where a change means
#: something. R-hat carries the separate job of saying whether the cell is
#: worth reading at all, which is the absolute limit below.
GATED = {
    "ess_bulk_min_per_sweep": "smaller",
    "ess_tail_min_per_sweep": "smaller",
    "rmse": "larger",
}


#: R-hat above this and the chains have not converged at all. Well above
#: the 1.01 reporting threshold: a cell sitting near that threshold flips
#: from run to run, and a gate that flips is noise. A real loss of
#: convergence is caught here or by the effective-sample-size gate above.
RHAT_LIMIT = 1.05


def failures(table: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    """Return the gate failures in `table`, empty when the run passes."""
    out = []
    unconverged = new[(new["metric"] == "rhat_max") & (new["value"] > RHAT_LIMIT)]
    for _, row in unconverged.iterrows():
        out.append(
            f"{row['cell']} rhat_max: {row['value']:.3f}, above {RHAT_LIMIT}; "
            "the chains have not converged and the cell's efficiency numbers "
            "describe nothing"
        )
    for _, row in table[table["separated"]].iterrows():
        direction = GATED.get(row["metric"])
        if direction is None:
            continue
        worse = row["ci_high"] < 1.0 if direction == "smaller" else row["ci_low"] > 1.0
        if worse:
            out.append(
                f"{row['cell']} {row['metric']}: ratio {row['ratio']:.3f} "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--separated-only",
        action="store_true",
        help="print only the metrics whose interval excludes one",
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="exit non-zero on an unconverged cell or a separated adverse move",
    )
    args = parser.parse_args()

    new = pd.read_csv(args.new)
    baseline = pd.read_csv(args.baseline)
    table = compare(baseline, new)
    shown = table[table["separated"]] if args.separated_only else table
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(shown.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
    if args.gate:
        problems = failures(table, new)
        if problems:
            print("\ngate failures:")
            for problem in problems:
                print(f"  {problem}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
