"""The parity gate against upstream, before any timing counts.

Upstream's cell-count prior thins by 1 / (b + 1), so its nominal
lambda_c = 25 corresponds to roughly five effective cells: two
implementations agreeing on the parameter do not agree on the model. The
realised cells per tessellation must match before a speed claim against
upstream means anything, and so must the posterior summaries at the same
configuration. Same posterior proven first, faster claimed second.

    python benchmarks/comparators/parity.py target/comparators/comparison.csv

Exits non-zero when a pair fails. A failure is not a slow method; it is
two methods fitting different models, and every timing ratio in that row
is void.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: The realised cells per tessellation may differ by this share before the
#: two are fitting different models. Wide, because the realised count is
#: itself a posterior mean with Monte Carlo error.
CELL_TOLERANCE = 0.15

#: Held-out summaries agree within this many standard errors of their
#: difference. Four, as the upstream fixture comparison in the crate's
#: tests uses, which is a two-sided 6e-5 level per summary.
SUMMARY_K = 4.0

#: The summaries compared. They are the ones both implementations produce
#: on the same held-out rows.
SUMMARIES = ("rmse", "lpd", "coverage_95", "width_95")


def failures(table: pd.DataFrame) -> list[str]:
    """Return the parity failures, empty when the gate passes."""
    out: list[str] = []
    # The gate reads the one-core rows: the same model on any core count,
    # but the timing it licenses is the one-core table's.
    if "cores" in table:
        table = table[table["cores"] == 1]
    ours = table[table["method"] == "thiessen"]
    theirs = table[table["method"] == "addivortes"]
    if ours.empty or theirs.empty:
        return ["no thiessen or addivortes cells in the table; nothing compared"]

    keys = ["dataset", "n", "p"]
    mine = ours.groupby(keys)
    yours = theirs.groupby(keys)
    for key, group in mine:
        if key not in yours.groups:
            out.append(f"{key}: upstream did not run this cell")
            continue
        other = yours.get_group(key)

        cells_a = group["cells_per_tessellation"].mean()
        cells_b = other["cells_per_tessellation"].mean()
        if np.isfinite(cells_a) and np.isfinite(cells_b) and cells_b > 0:
            relative = abs(cells_a - cells_b) / cells_b
            if relative > CELL_TOLERANCE:
                out.append(
                    f"{key} realised cells: {cells_a:.2f} against {cells_b:.2f}, "
                    f"{relative:.0%} apart"
                )

        for summary in SUMMARIES:
            a, b = group[summary].to_numpy(), other[summary].to_numpy()
            if not (np.isfinite(a).all() and np.isfinite(b).all()):
                continue
            if len(a) < 2 or len(b) < 2:
                continue
            se = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
            if se == 0.0:
                continue
            deviations = abs(a.mean() - b.mean()) / se
            if deviations > SUMMARY_K:
                out.append(
                    f"{key} {summary}: {a.mean():.4g} against {b.mean():.4g}, "
                    f"{deviations:.1f} standard errors apart"
                )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()

    problems = failures(pd.read_csv(args.csv))
    if not problems:
        print("parity holds; timings against upstream are readable")
        return
    print("parity failures:")
    for problem in problems:
        print(f"  {problem}")
    sys.exit(1)


if __name__ == "__main__":
    main()
