"""What every Python adapter shares: the metric formulas and the output.

The accuracy metrics are computed in the adapter rather than in the
harness because the posterior draws at every held-out row would be a
hundred megabytes a cell. The formulas are stated here and in
`common.R`, and the two must agree: a comparison where one method's log
predictive density is computed differently from another's is a comparison
of two formulas.

    rmse      sqrt(mean((mean_d f_di - y_i)^2))
    lpd       mean_i log(mean_d N(y_i; f_di, sigma_d))
    coverage  share of y_i inside the 2.5 and 97.5 per cent quantiles of
              one predictive sample per draw, f_di + sigma_d z_di
    width     the mean width of those intervals

The predictive sample is drawn from a generator fixed by the cell's seed,
so the interval is a function of the draws and nothing else.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

#: The offset that separates the predictive-sample generator from any
#: generator a method used, so the two cannot share a stream.
PREDICTIVE_SEED_OFFSET = 982_451_653


def accuracy(f: Array, sigma: Array, y: Array, seed: int) -> dict[str, float]:
    """Return the held-out metrics from per-draw f and sigma.

    Parameters
    ----------
    f : ndarray of shape (draws, rows)
        The posterior mean function at each held-out row.
    sigma : ndarray of shape (draws,)
        The residual standard deviation per draw.
    y : ndarray of shape (rows,)
        The held-out response.
    seed : int
        The cell's seed; keys the predictive sample.
    """
    mean = f.mean(axis=0)
    rmse = float(np.sqrt(np.mean((mean - y) ** 2)))

    scale = sigma[:, None]
    residual = (y[None, :] - f) / scale
    log_density = -0.5 * residual**2 - np.log(scale) - 0.5 * np.log(2.0 * np.pi)
    # log mean_d exp(.), by the log-sum-exp over draws.
    peak = log_density.max(axis=0)
    lpd = float(
        np.mean(peak + np.log(np.mean(np.exp(log_density - peak), axis=0)))
    )

    rng = np.random.default_rng(seed + PREDICTIVE_SEED_OFFSET)
    predictive = f + scale * rng.standard_normal(f.shape)
    lower, upper = np.quantile(predictive, [0.025, 0.975], axis=0)
    return {
        "rmse": rmse,
        "lpd": lpd,
        "coverage_95": float(np.mean((y >= lower) & (y <= upper))),
        "width_95": float(np.mean(upper - lower)),
    }


def write_draws(path: Path, series: dict[str, Array]) -> None:
    """Write the declared quantities as `chain,draw,quantity,value`.

    Each array is (chains, draws).
    """
    with path.open("w") as out:
        out.write("chain,draw,quantity,value\n")
        for name, values in series.items():
            chains, draws = values.shape
            for chain in range(chains):
                for draw in range(draws):
                    out.write(f"{chain},{draw},{name},{values[chain, draw]:.17e}\n")


def write_meta(path: Path, meta: dict) -> None:
    """Write the cell's metadata, with the environment it ran in."""
    meta = dict(meta)
    meta["platform"] = platform.platform()
    meta["processor"] = platform.processor()
    meta["python"] = platform.python_version()
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def read_csv(path: Path) -> tuple[Array, Array]:
    """Read a dataset written by `datasets.Dataset.write`."""
    table = np.loadtxt(path, delimiter=",", skiprows=1)
    return table[:, :-1], table[:, -1]
