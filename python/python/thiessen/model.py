"""The configuration and the fitted model.

`Model` holds a configuration and fits it; `FittedModel` holds the kept
draws and answers prediction and posterior queries.
"""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Union

import numpy as np
import numpy.typing as npt

from . import _native
from ._arrays import _as_design, _as_response
from ._config import FIELDS, _config_json
from ._seed import SeedLike, _resolve_seed

__all__ = ["FittedModel", "Model"]

MetricSpec = Sequence[Union[str, Mapping[str, Mapping[str, int]]]]


class Model:
    """An AddiVortes configuration.

    Every parameter left as `None` takes the core's default, stated below.
    The parameters are those of Stone and Gosling (2025), s. 2, with the
    sweep schedule that `fit` runs.

    Parameters
    ----------
    model : {'gaussian', 'probit', 'heteroscedastic'}, optional
        The observation model. Default 'gaussian'.
    m : int, optional
        Ensemble size m of the mean function. Default 200.
    nu : float, optional
        sigma^2 prior degrees of freedom nu. Default 6. The heteroscedastic
        model requires nu > 2.
    q : float, optional
        sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.
        Default 0.85.
    k : float, optional
        Cell-mean prior spread k: sigma_mu = 0.5 / (k sqrt(m)) on the
        response scaled to [-0.5, 0.5], or 3 / (k sqrt(m)) on the latent
        scale under the probit model (Chipman, George and McCulloch, 2010,
        s. 4). Default 3.
    sigma_c : float, optional
        Centre-coordinate prior and proposal standard deviation sigma_c in
        the scaled space. Default 0.8.
    omega : float, optional
        Dimension-count prior parameter omega; omega / p is the prior
        probability of including a covariate. Default min(3, p), resolved at
        fit. Must satisfy 0 < omega <= p.
    lambda_c : float, optional
        Cell-count prior rate lambda_c: b - 1 ~ Poisson(lambda_c). Default
        5, following AddiVortes >= 0.6.8. Stone and Gosling (2025), s. 2.3,
        report 25; pass ``lambda_c=25`` for the paper's setting.
    burn_in : int, optional
        Burn-in sweeps discarded. Default 200.
    draws : int, optional
        Posterior draws kept. Default 1000.
    thinning : int, optional
        Thinning interval; every `thinning`-th sweep after burn-in is kept.
        Default 1.
    prior_only : bool, optional
        Switch the likelihood off, so the chain draws from the prior and
        `predict` gives prior predictive draws. Default False.
    offset : float, optional
        Probit model only: the offset c in P(y = 1 | x) = Phi(c + f(x)).
        Default Phi^-1(ybar), resolved at fit.
    m_var : int, optional
        Heteroscedastic model only: the number m' of variance
        tessellations. Default 40.
    metric : sequence, optional
        The metric of each covariate column, one entry per column in column
        order: ``'euclidean'``, ``'categorical'``, or
        ``{'spherical': {'sphere': k}}`` with `k` the sphere label. Default
        Euclidean on every column. Non-Euclidean columns are not scaled.

    See Also
    --------
    FittedModel : The result of `fit`.

    Notes
    -----
    Stone, E. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
    Voronoi tessellations. *Journal of Computational and Graphical
    Statistics*, 34(3), 859-871.

    Examples
    --------
    >>> import numpy as np
    >>> from thiessen import Model
    >>> x = np.linspace(0.0, 1.0, 40).reshape(-1, 1)
    >>> y = 3.0 * x[:, 0] ** 2 - x[:, 0]
    >>> fitted = Model(m=10, burn_in=20, draws=30).fit(x, y, random_state=42)
    >>> fitted.predict(x).shape
    (40,)
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        m: int | None = None,
        nu: float | None = None,
        q: float | None = None,
        k: float | None = None,
        sigma_c: float | None = None,
        omega: float | None = None,
        lambda_c: float | None = None,
        burn_in: int | None = None,
        draws: int | None = None,
        thinning: int | None = None,
        prior_only: bool | None = None,
        offset: float | None = None,
        m_var: int | None = None,
        metric: MetricSpec | None = None,
    ) -> None:
        self.model = model
        self.m = m
        self.nu = nu
        self.q = q
        self.k = k
        self.sigma_c = sigma_c
        self.omega = omega
        self.lambda_c = lambda_c
        self.burn_in = burn_in
        self.draws = draws
        self.thinning = thinning
        self.prior_only = prior_only
        self.offset = offset
        self.m_var = m_var
        self.metric = metric

    def get_params(self) -> dict[str, Any]:
        """Return the set configuration fields.

        Returns
        -------
        dict
            Field name to value for every field, `None` for unset.
        """
        return {name: getattr(self, name) for name in FIELDS}

    def validate(self) -> None:
        """Validate the configuration without data.

        Raises
        ------
        ThiessenError
            Naming the field at fault. Checks that need the data, the
            omega <= p bound and the length of `metric`, run at fit.
        """
        _native.validate_config(_config_json(self.get_params()))

    def fit(
        self,
        X: Any,
        y: Any,
        random_state: SeedLike = None,
    ) -> FittedModel:
        """Fit the model to `X` and `y`.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design. Euclidean columns are min-max scaled over their
            training range; spherical columns are coordinates in radians and
            categorical columns are integer level codes, neither scaled.
        y : array_like of shape (n_samples,)
            The response. Labels in {0, 1} under the probit model.
        random_state : int, numpy.random.Generator, numpy.random.RandomState or None
            The seed. `None` draws fresh entropy. The resolved seed is on
            the returned object.

        Returns
        -------
        FittedModel
            The kept draws.

        Raises
        ------
        ThiessenError
            For an invalid configuration, or for data the core rejects:
            missing or non-finite values, a constant response, a constant
            column, fewer than two rows, or a row-count mismatch.
        """
        design = _as_design(X)
        response = _as_response(y)
        seed = _resolve_seed(random_state)
        fitted = _native.fit(_config_json(self.get_params()), design, response, seed)
        _emit_warnings(fitted, stacklevel=3)
        return FittedModel(fitted, seed)

    def __repr__(self) -> str:
        set_fields = ", ".join(
            f"{name}={value!r}"
            for name, value in self.get_params().items()
            if value is not None
        )
        return f"Model({set_fields})"


def _emit_warnings(fitted: _native.Fitted, stacklevel: int) -> None:
    """Re-raise the core's fit-time warnings as `UserWarning`."""
    for message in fitted.warnings:
        warnings.warn(message, UserWarning, stacklevel=stacklevel)


def _rebuild(payload: str, seed: int) -> FittedModel:
    """Reconstruct a `FittedModel` from its pickled state."""
    return FittedModel(_native.fitted_from_json(payload), seed)


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

    def __init__(self, fitted: _native.Fitted, seed: int) -> None:
        self._fitted = fitted
        self.random_state = seed

    @property
    def model(self) -> str:
        """str: The observation model."""
        return str(self._fitted.model)

    @property
    def config(self) -> dict[str, Any]:
        """dict: The resolved configuration, every field set."""
        parsed: dict[str, Any] = json.loads(self._fitted.config)
        return parsed

    @property
    def n_draws(self) -> int:
        """int: The number of kept draws."""
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
            model.
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
            f(x), or the latent mean c + f(x) under the probit model.
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
            Under the probit model, whose latent variance is one.
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
            model.
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
            Under the probit model, which has no continuous predictive
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
            Empty under the probit model, whose latent variance is one, and
            under the heteroscedastic model, where `predict_variance` gives
            s^2(x).
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
    def load(cls, path: str | os.PathLike[str], random_state: int = 0) -> FittedModel:
        """Read a fitted model from `path`.

        Parameters
        ----------
        path : str or os.PathLike
            A file written by `save`.
        random_state : int, default=0
            The seed to report on the loaded object, which the file does not
            carry.

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
        return cls(_native.fitted_from_json(payload), random_state)

    def __reduce__(self) -> tuple[Any, tuple[str, int]]:
        return _rebuild, (self._fitted.to_json(), self.random_state)

    def __repr__(self) -> str:
        return (
            f"FittedModel(model={self.model!r}, n_draws={self.n_draws}, "
            f"random_state={self.random_state})"
        )
