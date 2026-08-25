"""stochtree's BART sampler.

python -m adapters.stochtree_py <train.csv> <test.csv> <out-dir> <seed>
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.common import accuracy, read_csv, write_draws, write_meta  # noqa: E402
from cells import BURN_IN, CHAINS, DECLARED_ROWS, DRAWS, ENSEMBLE  # noqa: E402


def main() -> None:
    train, test, out, seed = sys.argv[1:5]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(seed)

    import stochtree
    from stochtree import BARTModel

    x, y = read_csv(Path(train))
    x_test, y_test = read_csv(Path(test))

    f_chains, sigma_chains = [], []
    fit_seconds = predict_seconds = 0.0
    for chain in range(CHAINS):
        model = BARTModel()
        started = time.perf_counter()
        model.sample(
            X_train=x,
            y_train=y,
            num_gfr=0,
            num_burnin=BURN_IN,
            num_mcmc=DRAWS,
            mean_forest_params={"num_trees": ENSEMBLE},
            general_params={"random_seed": seed + chain},
        )
        fit_seconds += time.perf_counter() - started
        started = time.perf_counter()
        # (rows, draws) from stochtree; the shared shape is (draws, rows).
        f_chains.append(np.asarray(model.predict(x_test)["y_hat"]).T)
        predict_seconds += time.perf_counter() - started
        sigma_chains.append(np.sqrt(np.asarray(model.global_var_samples)))

    f = np.concatenate(f_chains, axis=0)
    sigma = np.concatenate(sigma_chains, axis=0)

    declared = min(DECLARED_ROWS, f.shape[1])
    series = {f"f[{i}]": np.stack([c[:, i] for c in f_chains]) for i in range(declared)}
    series["sigma"] = np.stack(sigma_chains)
    write_draws(out_dir / "draws.csv", series)

    write_meta(
        out_dir / "meta.json",
        {
            "method": "stochtree",
            "version": getattr(stochtree, "__version__", "unknown"),
            "chains": CHAINS,
            "draws": DRAWS,
            "burn_in": BURN_IN,
            "ensemble": ENSEMBLE,
            "fit_seconds": fit_seconds,
            # stochtree does not report the two phases apart either, so
            # the split is apportioned by sweep count, as in `r_methods.R`.
            "warmup_seconds": fit_seconds * BURN_IN / (BURN_IN + DRAWS),
            "post_warmup_seconds": fit_seconds * DRAWS / (BURN_IN + DRAWS),
            "predict_seconds": predict_seconds,
            "cells_per_tessellation": None,
            **accuracy(f, sigma, y_test, seed),
        },
    )


if __name__ == "__main__":
    main()
