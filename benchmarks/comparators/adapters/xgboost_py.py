"""The accuracy baseline: gradient-boosted trees, no posterior.

    python -m adapters.xgboost_py <train.csv> <test.csv> <out-dir> <seed>

XGBoost has no posterior, so it has no effective sample size, no log
predictive density and no interval. It carries the held-out error alone,
and it is here because a Bayesian method that loses to it on accuracy has
a case to answer whatever its sampling efficiency.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.common import read_csv, write_meta  # noqa: E402

#: Rounds and depth are the library defaults for regression. Tuning the
#: baseline and not the comparators would make it a different claim.
ROUNDS = 200


def main() -> None:
    train, test, out, seed = sys.argv[1:5]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import xgboost

    x, y = read_csv(Path(train))
    x_test, y_test = read_csv(Path(test))

    model = xgboost.XGBRegressor(
        n_estimators=ROUNDS, random_state=int(seed), n_jobs=1
    )
    started = time.perf_counter()
    model.fit(x, y)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predictions = np.asarray(model.predict(x_test))
    predict_seconds = time.perf_counter() - started

    write_meta(
        out_dir / "meta.json",
        {
            "method": "xgboost",
            "version": xgboost.__version__,
            "chains": 0,
            "draws": 0,
            "burn_in": 0,
            "ensemble": ROUNDS,
            "fit_seconds": fit_seconds,
            "warmup_seconds": 0.0,
            "post_warmup_seconds": 0.0,
            "predict_seconds": predict_seconds,
            "rmse": float(np.sqrt(np.mean((predictions - y_test) ** 2))),
            "lpd": None,
            "coverage_95": None,
            "width_95": None,
            "cells_per_tessellation": None,
        },
    )


if __name__ == "__main__":
    main()
