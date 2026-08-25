"""This library, through its Python package.

python -m adapters.thiessen_py <train.csv> <test.csv> <out-dir> <seed>
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

    import thiessen
    from thiessen import TermParams
    from thiessen.sampler import Sampler

    x, y = read_csv(Path(train))
    x_test, y_test = read_csv(Path(test))

    f_chains, sigma_chains = [], []
    warmup = post = predict = 0.0
    for chain in range(CHAINS):
        # The schedule is driven here so warm-up and sampling are timed
        # apart: the per-iteration cost a published number quotes excludes
        # initialisation and burn-in.
        start = time.perf_counter()
        sampler = Sampler(
            x,
            y,
            mean_params=TermParams(tessellations=ENSEMBLE),
            random_state=seed + chain,
        )
        sampler.step(BURN_IN)
        warmup += time.perf_counter() - start
        start = time.perf_counter()
        for _ in range(DRAWS):
            sampler.step()
            sampler.keep()
        post += time.perf_counter() - start
        fitted = sampler.finish()

        start = time.perf_counter()
        draws = np.asarray(fitted.predict_draws(x_test))
        predict += time.perf_counter() - start
        f_chains.append(draws)
        sigma_chains.append(np.asarray(fitted.sigma()))

    f = np.concatenate(f_chains, axis=0)
    sigma = np.concatenate(sigma_chains, axis=0)

    declared = min(DECLARED_ROWS, f.shape[1])
    series = {f"f[{i}]": np.stack([c[:, i] for c in f_chains]) for i in range(declared)}
    series["sigma"] = np.stack(sigma_chains)
    write_draws(out_dir / "draws.csv", series)

    write_meta(
        out_dir / "meta.json",
        {
            "method": "thiessen",
            "version": thiessen.CORE_VERSION,
            "chains": CHAINS,
            "draws": DRAWS,
            "burn_in": BURN_IN,
            "ensemble": ENSEMBLE,
            "warmup_seconds": warmup,
            "post_warmup_seconds": post,
            "predict_seconds": predict,
            "fit_seconds": warmup + post,
            # The realised structure, for the parity gate against upstream:
            # upstream's cell-count prior thins by 1 / (b + 1), so its
            # nominal lambda is not the count it realises.
            "cells_per_tessellation": float(np.mean(fitted.cell_counts())),
            **accuracy(f, sigma, y_test, seed),
        },
    )


if __name__ == "__main__":
    main()
