"""This library, through its Python package.

python -m adapters.thiessen_py <train.csv> <test.csv> <out-dir> <seed>

On the main and scaling grids the chains are driven one at a time
through `Sampler`, so warm-up and sampling are timed apart. On the cores
grid they run together through `Model.fit(n_chains=, n_threads=)` at
every core count, one included, one chain per thread, which is the path
a user who sets a thread count gets and the one code path the scaling
across core counts is read along; the per-phase split is then
apportioned by sweep count, as the R adapters do for every method.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adapters.common import accuracy, read_csv, write_draws, write_meta  # noqa: E402
from cells import (  # noqa: E402
    BURN_IN,
    CHAINS,
    DECLARED_ROWS,
    DRAWS,
    ENSEMBLE,
    GRID,
    THREADS,
)


def chains_in_turn(x, y, x_test, seed):
    """One chain at a time through `Sampler`, warm-up timed apart.

    Returns the per-chain f and sigma, the phase times, and the last
    chain's fitted model for the realised structure.
    """
    from thiessen import TermParams
    from thiessen.sampler import Sampler

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
    return f_chains, sigma_chains, warmup, post, predict, fitted


def chains_together(x, y, x_test, seed):
    """Every chain at once through `Model.fit`, one chain per thread.

    The chain seeds are the core's `chain_seed(seed, k)` rather than the
    `seed + k` of the one-at-a-time path, so the two paths draw different
    chains of the same posterior; neither is the other's reference. The
    pooled fit stacks the chains in order, so the draws reshape to
    (chains, draws, ...) without a reordering.

    A discarded fit at a trivial schedule runs first, so the one-off
    costs of the first multi-chain fit in a process (the diagnostics
    import arviz on first use, about two seconds) fall outside the timed
    region, as the harness's discarded first cell keeps them out of the
    run.
    """
    import warnings

    from thiessen import Model, TermParams

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Model(mean_params=TermParams(tessellations=2), burn_in=1, draws=2).fit(
            x[:20], y[:20], random_state=seed, n_chains=2, n_threads=THREADS
        )

    model = Model(
        mean_params=TermParams(tessellations=ENSEMBLE), burn_in=BURN_IN, draws=DRAWS
    )
    start = time.perf_counter()
    fitted = model.fit(x, y, random_state=seed, n_chains=CHAINS, n_threads=THREADS)
    fit_seconds = time.perf_counter() - start
    # The fit does not report its phases apart, so the split is
    # apportioned by sweep count, as in `r_methods.R`.
    warmup = fit_seconds * BURN_IN / (BURN_IN + DRAWS)
    post = fit_seconds * DRAWS / (BURN_IN + DRAWS)

    start = time.perf_counter()
    draws = np.asarray(fitted.predict_draws(x_test))
    predict = time.perf_counter() - start
    f_chains = list(draws.reshape(CHAINS, DRAWS, -1))
    sigma_chains = list(np.asarray(fitted.sigma()).reshape(CHAINS, DRAWS))
    return f_chains, sigma_chains, warmup, post, predict, fitted


def main() -> None:
    train, test, out, seed = sys.argv[1:5]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(seed)

    import thiessen

    x, y = read_csv(Path(train))
    x_test, y_test = read_csv(Path(test))

    run = chains_together if GRID == "cores" else chains_in_turn
    f_chains, sigma_chains, warmup, post, predict, fitted = run(x, y, x_test, seed)

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
            "threads": THREADS,
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
