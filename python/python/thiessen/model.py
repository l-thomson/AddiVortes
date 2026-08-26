"""The configuration and the fitted model.

`Model` holds a configuration and fits it; `FittedModel` holds the kept
draws and answers prediction and posterior queries.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from . import _native
from ._arrays import _as_design, _as_response
from ._config import _config_json
from ._convergence import _warn_convergence
from ._inference import _to_inference_data
from ._params import Outcome, Params
from ._seed import SeedLike, _resolve_seed
from .params import TermParams

__all__ = ["FittedModel", "Model"]


class Model(Params):
    """An AddiVortes configuration.

    The outcome family and the two ensembles are objects with the names of
    the stored configuration; parameters left unset take the core's
    defaults. The parameters are those of Stone and Gosling (2025), s. 2,
    with the sweep schedule that `fit` runs.

    Parameters
    ----------
    outcome : Outcome, optional
        The outcome family, from `gaussian`, `probit` or one of the
        experimental constructors of `thiessen.families`. Default
        ``gaussian()``.
    mean_params : TermParams, optional
        The ensemble describing the average. Default ``TermParams()``,
        whose tessellation count resolves to 200.
    variance_params : TermParams, optional
        The ensemble describing the spread. Default none: the spread is
        constant. A positive tessellation count selects the
        heteroscedastic model and needs the Gaussian family with nu > 2;
        the paper's count is 40.
    burn_in : int, default=200
        Burn-in sweeps discarded.
    draws : int, default=1000
        Posterior draws kept.
    thinning : int, default=1
        Thinning interval; every `thinning`-th sweep after burn-in is
        kept.
    prior_only : bool, default=False
        Switch the likelihood off, so the chain draws from the prior and
        `predict` gives prior predictive draws.

    See Also
    --------
    FittedModel : The result of `fit`.
    thiessen.gaussian, thiessen.probit : The outcome families.
    thiessen.TermParams : One ensemble's parameters.

    Notes
    -----
    Stone, E. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
    Voronoi tessellations. *Journal of Computational and Graphical
    Statistics*, 34(3), 859-871.

    Examples
    --------
    >>> import numpy as np
    >>> from thiessen import Model, TermParams
    >>> x = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    >>> y = 3.0 * x[:, 0] ** 2 - x[:, 0]
    >>> model = Model(mean_params=TermParams(tessellations=10),
    ...               burn_in=20, draws=30)
    >>> model.fit(x, y, random_state=42).predict(x).shape
    (40,)

    The heteroscedastic model attaches a variance ensemble:

    >>> hetero = Model(variance_params=TermParams(tessellations=40))
    """

    def __init__(
        self,
        *,
        outcome: Outcome | None = None,
        mean_params: TermParams | None = None,
        variance_params: TermParams | None = None,
        burn_in: int = 200,
        draws: int = 1000,
        thinning: int = 1,
        prior_only: bool = False,
    ) -> None:
        self.outcome = outcome
        self.mean_params = mean_params
        self.variance_params = variance_params
        self.burn_in = burn_in
        self.draws = draws
        self.thinning = thinning
        self.prior_only = prior_only

    def _json(self) -> str:
        return _config_json(
            self.outcome,
            self.mean_params,
            self.variance_params,
            self.burn_in,
            self.draws,
            self.thinning,
            self.prior_only,
        )

    def validate(self) -> None:
        """Validate the configuration without data.

        Raises
        ------
        ThiessenError
            Naming the field at fault. Checks that need the data, the
            omega <= p bound and the length of ``metric``, run at fit.
        TypeError
            For a group that is not its parameter object.
        """
        _native.validate_config(self._json())

    def fit(
        self,
        X: Any,
        y: Any,
        random_state: SeedLike = None,
        n_chains: int = 1,
        n_threads: int = 1,
    ) -> FittedModel:
        """Fit the model to `X` and `y`.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design. Euclidean columns are min-max scaled over their
            training range; spherical columns are coordinates in radians
            and categorical columns are integer level codes, neither
            scaled.
        y : array_like of shape (n_samples,)
            The response. Labels in {0, 1} under the probit family.
        random_state : int, numpy.random.Generator, numpy.random.RandomState or None
            The seed. `None` draws fresh entropy. The resolved seed is on
            the returned object.
        n_chains : int, default=1
            The number of chains to run. Each chain has its own seed,
            derived from the resolved seed in the core, and the draws of
            the chains are pooled.
        n_threads : int, default=1
            The number of threads. The chains are spread over at most this
            many threads, each chain on one thread with its own generator,
            so the draws do not depend on it; the returned model splits
            the rows of a prediction over the same number.

        Returns
        -------
        FittedModel
            The kept draws of every chain.

        Warns
        -----
        UserWarning
            Where two or more chains ran and R-hat exceeds 1.01 or an
            effective sample size falls below 400 (Vehtari and others,
            2021), or where arviz is not installed to compute them.

        Raises
        ------
        ThiessenError
            For an invalid configuration, or for data the core rejects:
            missing or non-finite values, a constant response, a constant
            column, fewer than two rows, or a row-count mismatch.
        ValueError
            If `n_chains` or `n_threads` is not a positive integer.
        """
        design = _as_design(X)
        response = _as_response(y)
        seed = _resolve_seed(random_state)
        chains = _resolve_chains(n_chains)
        threads = _resolve_threads(n_threads)
        fitted = _native.fit(self._json(), design, response, seed, chains, threads)
        _emit_warnings(fitted, stacklevel=3)
        _warn_convergence(fitted, chains, design, stacklevel=3)
        return FittedModel(fitted, seed, chains, threads)


def _emit_warnings(fitted: _native.Fitted, stacklevel: int) -> None:
    """Re-raise the core's fit-time warnings as `UserWarning`."""
    for message in fitted.warnings:
        warnings.warn(message, UserWarning, stacklevel=stacklevel)


def _resolve_chains(n_chains: Any) -> int:
    """Return `n_chains` as a positive integer."""
    chains = int(n_chains)
    if chains != n_chains or chains < 1:
        raise ValueError(f"n_chains must be a positive integer; got {n_chains!r}")
    return chains


def _resolve_threads(n_threads: Any) -> int:
    """Return `n_threads` as a positive integer."""
    threads = int(n_threads)
    if threads != n_threads or threads < 1:
        raise ValueError(f"n_threads must be a positive integer; got {n_threads!r}")
    return threads


def _rebuild(
    payload: str, seed: int, n_chains: int = 1, n_threads: int = 1
) -> FittedModel:
    """Reconstruct a `FittedModel` from its pickled state."""
    return FittedModel(
        _native.fitted_from_json(payload, n_threads), seed, n_chains, n_threads
    )


class FittedModel:
    """A fitted model: the kept draws and the queries over them.

    Returned by `Model.fit`; not constructed directly. Prediction methods
    take a design with the column count of the fit and return values on the
    caller's scale.

    Attributes
    ----------
    random_state : int
        The resolved seed of the fit.

    See Also
    --------
    Model : The configuration.
    """

    def __init__(
        self,
        fitted: _native.Fitted,
        seed: int,
        n_chains: int = 1,
        n_threads: int = 1,
    ) -> None:
        self._fitted = fitted
        self.random_state = seed
        self._n_chains = n_chains
        self._n_threads = n_threads

    @property
    def n_chains(self) -> int:
        """int: The number of chains the draws were pooled from."""
        return self._n_chains

    @property
    def n_threads(self) -> int:
        """int: The number of threads a prediction splits its rows over.

        Settable: a positive integer, in force from the next prediction.
        """
        return self._n_threads

    @n_threads.setter
    def n_threads(self, n_threads: int) -> None:
        threads = _resolve_threads(n_threads)
        self._fitted.set_threads(threads)
        self._n_threads = threads

    @property
    def model(self) -> str:
        """str: The observation model."""
        return str(self._fitted.model)

    @property
    def config(self) -> dict[str, Any]:
        """dict: The resolved configuration, the core's four groups.

        Every field is set: the outcome family under ``outcome``, the two
        ensembles under ``mean_params`` and ``variance_params``, and the
        sweep schedule under ``general_params``.
        """
        parsed: dict[str, Any] = json.loads(self._fitted.config)
        return parsed

    @property
    def n_draws(self) -> int:
        """int: The number of kept draws over every chain."""
        return int(self._fitted.n_draws)

    @property
    def in_sample_rmse(self) -> float:
        """float: The in-sample root mean squared error.

        The root mean squared error of the posterior mean at the training
        rows.
        """
        return float(self._fitted.in_sample_rmse)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Tuple of str: the fit-time warnings."""
        return tuple(self._fitted.warnings)

    def predict(self, X: Any) -> npt.NDArray[np.float64]:
        """Return the posterior mean at each row of `X`.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        numpy.ndarray of shape (n_samples,)
            The posterior mean of f(x), or of P(y = 1 | x) under the probit
            family.
        """
        return self._fitted.predict(_as_design(X))

    def predict_draws(self, X: Any) -> npt.NDArray[np.float64]:
        """Return the quantity of `predict` for every kept draw.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        numpy.ndarray of shape (n_draws, n_samples)
            Draw-major.
        """
        return self._fitted.predict_draws(_as_design(X))

    def predict_latent(self, X: Any) -> npt.NDArray[np.float64]:
        """Return the mean function for every kept draw.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        numpy.ndarray of shape (n_draws, n_samples)
            f(x), or the latent mean c + f(x) under the probit family.
        """
        return self._fitted.predict_latent(_as_design(X))

    def predict_variance(self, X: Any) -> npt.NDArray[np.float64]:
        """Return the variance of y given f for every kept draw.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        numpy.ndarray of shape (n_draws, n_samples)
            sigma^2 under the Gaussian model, constant across rows; s^2(x)
            under the heteroscedastic model.

        Raises
        ------
        ThiessenError
            Under the probit family, whose latent variance is one.
        """
        return self._fitted.predict_variance(_as_design(X))

    def predict_quantiles(
        self, X: Any, probs: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        """Return posterior quantiles of the quantity of `predict`.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        probs : sequence of float
            Probabilities in (0, 1).

        Returns
        -------
        numpy.ndarray of shape (n_samples, len(probs))
            Type 7 interpolation over the kept draws.

        Raises
        ------
        ThiessenError
            For an empty `probs` or a probability outside (0, 1).
        """
        return self._fitted.predict_quantiles(_as_design(X), [float(p) for p in probs])

    def credible_interval(self, X: Any, level: float = 0.95) -> npt.NDArray[np.float64]:
        """Return the central credible interval for the mean at `level`.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        level : float, optional
            Interval level in (0, 1). Default 0.95.

        Returns
        -------
        numpy.ndarray of shape (n_samples, 2)
            Lower and upper ends. On the probability scale under the probit
            family.
        """
        return self._fitted.credible_interval(_as_design(X), float(level))

    def prediction_interval(
        self, X: Any, level: float = 0.95
    ) -> npt.NDArray[np.float64]:
        """Return the central predictive interval for a new observation.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        level : float, optional
            Interval level in (0, 1). Default 0.95.

        Returns
        -------
        numpy.ndarray of shape (n_samples, 2)
            Quantiles of the equal-weight mixture over kept draws of
            N(f_d(x), s_d^2(x)).

        Raises
        ------
        ThiessenError
            Under the probit family, which has no continuous predictive
            distribution.
        """
        return self._fitted.prediction_interval(_as_design(X), float(level))

    def log_likelihood(self, X: Any, y: Any) -> npt.NDArray[np.float64]:
        """Return the pointwise log-likelihood per draw.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        y : array_like of shape (n_samples,)
            The response.

        Returns
        -------
        numpy.ndarray of shape (n_draws, n_samples)
            Draw-major.
        """
        return self._fitted.log_likelihood(_as_design(X), _as_response(y))

    def sigma(self) -> npt.NDArray[np.float64]:
        """Return sigma per kept draw.

        Returns
        -------
        numpy.ndarray of shape (n_draws,)
            Empty under the probit family, whose latent variance is one,
            and under the heteroscedastic model, where `predict_variance`
            gives s^2(x).
        """
        return self._fitted.sigma()

    def cell_counts(self) -> npt.NDArray[np.float64]:
        """Return the mean cells per mean tessellation, per kept draw.

        Returns
        -------
        numpy.ndarray of shape (n_draws,)
            The mean over the m tessellations of each draw.
        """
        return self._fitted.cell_counts()

    def dimension_counts(self) -> npt.NDArray[np.float64]:
        """Return the mean active covariates per mean tessellation.

        Returns
        -------
        numpy.ndarray of shape (n_draws,)
            The mean over the m tessellations of each draw.
        """
        return self._fitted.dimension_counts()

    def variable_inclusion_proportions(self) -> npt.NDArray[np.float64]:
        """Return the share of active dimensions falling on each covariate.

        Returns
        -------
        numpy.ndarray of shape (n_features,)
            Sums to one (Chipman, George and McCulloch, 2010, s. 5.1).
        """
        return self._fitted.variable_inclusion_proportions()

    def to_inference_data(self, X: Any, y: Any) -> Any:
        """Return the fit as an arviz `DataTree`.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The rows the observation dimension indexes, usually the training
            design.
        y : array_like of shape (n_samples,)
            The observed response.

        Returns
        -------
        xarray.DataTree
            The `posterior` group carries `mu`, the mean function per draw,
            with `sigma` under the Gaussian model only, and the per-draw cell
            and dimension counts; `posterior_predictive` and `log_likelihood`
            carry `y`; `observed_data` carries the response.

        Raises
        ------
        ImportError
            If arviz is not installed.
        ValueError
            If `X` and `y` disagree on the number of rows.

        Notes
        -----
        The chain dimension of every group holds the chains of the fit. The
        predictive replicates are drawn in numpy from the fit's resolved
        seed rather than by the core.
        """
        return _to_inference_data(
            self._fitted,
            _as_design(X),
            _as_response(y),
            self.random_state,
            self._n_chains,
        )

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write the fitted model to `path`.

        Parameters
        ----------
        path : str or os.PathLike
            The destination. The format is the core's serde representation,
            which reloads bit-exact.

        Raises
        ------
        OSError
            If the file cannot be written.
        """
        with open(os.fspath(path), "w", encoding="utf-8") as handle:
            handle.write(self._fitted.to_json())

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        random_state: int = 0,
        n_chains: int = 1,
        n_threads: int = 1,
    ) -> FittedModel:
        """Read a fitted model from `path`.

        Parameters
        ----------
        path : str or os.PathLike
            A file written by `save`.
        random_state : int, default=0
            The seed to report on the loaded object, which the file does not
            carry.
        n_chains : int, default=1
            The number of chains the draws were pooled from, which the file
            does not carry.
        n_threads : int, default=1
            The number of threads a prediction splits its rows over, which
            the file does not carry.

        Returns
        -------
        FittedModel
            The loaded model.

        Raises
        ------
        OSError
            If the file cannot be read.
        ThiessenError
            If the contents are not a fitted model of this crate version.
        """
        with open(os.fspath(path), encoding="utf-8") as handle:
            payload = handle.read()
        threads = _resolve_threads(n_threads)
        return cls(
            _native.fitted_from_json(payload, threads), random_state, n_chains, threads
        )

    def __reduce__(self) -> tuple[Any, tuple[str, int, int, int]]:
        return _rebuild, (
            self._fitted.to_json(),
            self.random_state,
            self._n_chains,
            self._n_threads,
        )

    def __repr__(self) -> str:
        return (
            f"FittedModel(model={self.model!r}, n_chains={self.n_chains}, "
            f"n_draws={self.n_draws}, random_state={self.random_state})"
        )
