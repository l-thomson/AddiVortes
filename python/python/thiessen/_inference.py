"""Conversion of a fitted model to an arviz `DataTree`.

Requires arviz, the `arviz` extra. The groups follow the PyMC and numpyro
convention: `posterior`, `posterior_predictive`, `log_likelihood` and
`observed_data`, over the single chain the sampler runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from . import _native

__all__ = ["_to_inference_data"]


def _replicates(
    fitted: _native.Fitted,
    design: npt.NDArray[np.float64],
    seed: int,
) -> npt.NDArray[np.float64]:
    """Draw one posterior predictive replicate per kept draw.

    The replicates are drawn in numpy from a generator seeded by the fit's
    resolved seed, not by the core, so they are reproducible but are not the
    core's draws.
    """
    rng = np.random.default_rng(seed)
    draws = fitted.predict_draws(design)
    if fitted.model == "probit":
        return rng.binomial(1, np.clip(draws, 0.0, 1.0)).astype(np.float64)
    variance = fitted.predict_variance(design)
    return draws + rng.normal(0.0, np.sqrt(variance))


def _to_inference_data(
    fitted: _native.Fitted,
    design: npt.NDArray[np.float64],
    response: npt.NDArray[np.float64],
    seed: int,
) -> Any:
    """Build the `DataTree` of a fitted model over `design` and `response`.

    Parameters
    ----------
    fitted : Fitted
        The native handle.
    design : numpy.ndarray of shape (n_samples, n_features)
        The rows the observation dimension indexes.
    response : numpy.ndarray of shape (n_samples,)
        The observed response.
    seed : int
        The fit's resolved seed, which seeds the predictive replicates.

    Returns
    -------
    xarray.DataTree
        The `posterior`, `posterior_predictive`, `log_likelihood` and
        `observed_data` groups.

    Raises
    ------
    ImportError
        If arviz is not installed.
    ValueError
        If `design` and `response` disagree on the number of rows.
    """
    try:
        import arviz as az
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "to_inference_data needs arviz; install the `arviz` extra"
        ) from error

    if design.shape[0] != response.shape[0]:
        raise ValueError(
            f"X has {design.shape[0]} rows and y has {response.shape[0]}; "
            "the observation dimension needs one label per row"
        )

    latent = fitted.predict_latent(design)
    posterior: dict[str, npt.NDArray[np.float64]] = {
        "mu": latent[np.newaxis, :, :],
        "cell_count": np.asarray(fitted.cell_counts())[np.newaxis, :],
        "dimension_count": np.asarray(fitted.dimension_counts())[np.newaxis, :],
    }
    sigma = np.asarray(fitted.sigma())
    # Empty under the probit model, whose latent variance is one, and under
    # the heteroscedastic model, whose variance varies with x.
    if sigma.size:
        posterior["sigma"] = sigma[np.newaxis, :]

    groups = {
        "posterior": posterior,
        "posterior_predictive": {
            "y": _replicates(fitted, design, seed)[np.newaxis, :, :]
        },
        "log_likelihood": {
            "y": np.asarray(fitted.log_likelihood(design, response))[np.newaxis, :, :]
        },
        "observed_data": {"y": response},
    }

    return az.from_dict(
        groups,
        coords={"observation": np.arange(design.shape[0])},
        dims={"mu": ["observation"], "y": ["observation"]},
    )
