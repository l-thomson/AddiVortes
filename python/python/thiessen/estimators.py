"""scikit-learn estimators over the core.

`AddiVortesRegressor` fits the Gaussian and heteroscedastic models;
`AddiVortesClassifier` fits the binary probit model. Both meet the
scikit-learn estimator contract, so they compose with `Pipeline`,
`GridSearchCV`, `cross_val_score` and `sklearn.inspection`.

Requires scikit-learn 1.6 or later, the `sklearn` extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import type_of_target, unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from . import _native
from ._arrays import _as_response
from ._config import FIELDS, _config_json
from ._encoding import Encoding, columns_of, resolve_mask
from ._seed import SeedLike, _resolve_seed

__all__ = ["AddiVortesClassifier", "AddiVortesRegressor"]

MetricEntry = Union[str, "dict[str, dict[str, int]]"]

#: Parameters that are not fields of the core's configuration.
_NON_CONFIG = frozenset({"random_state", "categorical_features"})


_COMMON_DOC = """    m : int, default=200
        Ensemble size m of the mean function.
    nu : float, default=6.0
        sigma^2 prior degrees of freedom nu.
    q : float, default=0.85
        sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.
    k : float, default=3.0
        Cell-mean prior spread k: sigma_mu = 0.5 / (k sqrt(m)) on the response
        scaled to [-0.5, 0.5], or 3 / (k sqrt(m)) on the latent scale under
        the probit model (Chipman, George and McCulloch, 2010, s. 4).
    sigma_c : float, default=0.8
        Centre-coordinate prior and proposal standard deviation sigma_c.
    omega : float, default=None
        Dimension-count prior parameter omega; omega / p is the prior
        probability of including a covariate. `None` resolves to
        min(3, n_features_in_) at fit, so the default is valid for any input.
    lambda_c : float, default=5.0
        Cell-count prior rate lambda_c: b - 1 ~ Poisson(lambda_c). The default
        follows AddiVortes >= 0.6.8. Stone and Gosling (2025), s. 2.3, report
        25; pass ``lambda_c=25`` for the paper's setting.
    burn_in : int, default=200
        Burn-in sweeps discarded.
    draws : int, default=1000
        Posterior draws kept.
    thinning : int, default=1
        Every `thinning`-th sweep after burn-in is kept.
    prior_only : bool, default=False
        Switch the likelihood off, so the chain draws from the prior.
    metric : sequence, default=None
        The metric of each input column: ``'euclidean'``, ``'categorical'``,
        or ``{'spherical': {'sphere': k}}``. `None` is Euclidean throughout.
    categorical_features : None, 'from_dtype' or array_like, default=None
        The categorical columns. `None` takes the input as numeric; for a
        column that needs encoding, either name it here or encode it yourself
        with ``OneHotEncoder(drop='first')`` in a `ColumnTransformer`.
        'from_dtype' reads the pandas categorical dtypes; an array of indices
        or a boolean mask names the columns. A named column becomes d - 1
        treatment-contrast indicators, the first level as reference, unless its
        `metric` entry is ``'categorical'``, in which case it passes as
        integer level codes.
    random_state : int, Generator, RandomState or None, default=None
        The seed. An integer passes through to the core unchanged, so
        `AddiVortesRegressor(random_state=1)` and `Model(random_state=1)` draw
        alike."""


def _with_common(cls: type) -> type:
    """Substitute the shared parameter documentation into a docstring."""
    cls.__doc__ = (cls.__doc__ or "").replace("{common}", _COMMON_DOC)
    return cls


class _BaseAddiVortes(BaseEstimator):
    """Shared configuration, fitting and prediction."""

    def __init__(
        self,
        *,
        m: int = 200,
        nu: float = 6.0,
        q: float = 0.85,
        k: float = 3.0,
        sigma_c: float = 0.8,
        omega: float | None = None,
        lambda_c: float = 5.0,
        burn_in: int = 200,
        draws: int = 1000,
        thinning: int = 1,
        prior_only: bool = False,
        metric: Sequence[MetricEntry] | None = None,
        categorical_features: str | Sequence[Any] | None = None,
        random_state: SeedLike = None,
    ) -> None:
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
        self.metric = metric
        self.categorical_features = categorical_features
        self.random_state = random_state

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = False
        tags.target_tags.required = True
        return tags

    def _config(self, model: str, core_metric: list[MetricEntry]) -> str:
        params = {
            name: getattr(self, name, None)
            for name in FIELDS
            if name not in {"model", "metric"}
        }
        params["model"] = model
        varies = any(entry != "euclidean" for entry in core_metric)
        params["metric"] = core_metric if varies else None
        return _config_json(params)

    def _encode_fit(self, x: Any) -> npt.NDArray[np.float64]:
        """Learn the categorical encoding and return the core's design."""
        columns = columns_of(x)
        mask = resolve_mask(self.categorical_features, x, len(columns))
        self._encoding = Encoding.fit(columns, mask, self.metric)
        return self._encoding.transform(columns)

    def _encode_predict(self, x: Any) -> npt.NDArray[np.float64]:
        columns = columns_of(x)
        if len(columns) != self.n_features_in_:
            raise ValueError(
                f"X has {len(columns)} features, but this "
                f"{type(self).__name__} is expecting {self.n_features_in_} "
                "features as input"
            )
        return self._encoding.transform(columns)

    def _validate_fit(self, x: Any, y: Any) -> tuple[npt.NDArray[np.float64], Any]:
        """Validate and encode, setting `n_features_in_`."""
        if self.categorical_features is None:
            x_checked, y_checked = validate_data(
                self, x, y, dtype=np.float64, ensure_min_samples=2
            )
            self._encoding = Encoding.fit(
                columns_of(x_checked),
                np.zeros(x_checked.shape[1], dtype=bool),
                self.metric,
            )
            return x_checked, y_checked
        names = getattr(x, "columns", None)
        if names is not None:
            self.feature_names_in_ = np.asarray(names, dtype=object)
        self.n_features_in_ = len(columns_of(x))
        return self._encode_fit(x), np.asarray(y)

    def _validate_predict(self, x: Any) -> npt.NDArray[np.float64]:
        check_is_fitted(self)
        if self.categorical_features is None:
            checked = validate_data(self, x, reset=False, dtype=np.float64)
            return np.asarray(checked, dtype=np.float64)
        return self._encode_predict(x)

    def _fit_core(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        seed = _resolve_seed(self.random_state)
        self.random_state_ = seed
        self._fitted: _native.Fitted = _native.fit(
            self._config(self._model_name(), self._encoding.core_metric),
            np.ascontiguousarray(x, dtype=np.float64),
            _as_response(y),
            seed,
        )

    def _model_name(self) -> str:
        raise NotImplementedError

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        fitted = state.pop("_fitted", None)
        if fitted is not None:
            state["_fitted_json"] = fitted.to_json()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        payload = state.pop("_fitted_json", None)
        self.__dict__.update(state)
        if payload is not None:
            self._fitted = _native.fitted_from_json(payload)


@_with_common
class AddiVortesRegressor(RegressorMixin, _BaseAddiVortes):
    """AddiVortes regression.

        Bayesian regression on a sum of Voronoi tessellations (Stone and Gosling,
        2025). The posterior mean of f(x) is the prediction.

        Parameters
        ----------
        model : {'gaussian', 'heteroscedastic'}, default='gaussian'
            The observation model. 'gaussian' draws one sigma^2 per sweep;
            'heteroscedastic' carries an ensemble of variance tessellations, so
            the residual variance varies with x. Binary responses need
            `AddiVortesClassifier`.
        m_var : int, default=40
            Heteroscedastic model only: the number m' of variance tessellations.
    {common}

        Attributes
        ----------
        n_features_in_ : int
            Number of features seen at fit.
        feature_names_in_ : ndarray of shape (n_features_in_,)
            Feature names seen at fit, when the input carried them.
        random_state_ : int
            The resolved seed.

        See Also
        --------
        AddiVortesClassifier : The binary probit model.

        Notes
        -----
        Partial dependence and individual conditional expectation come from
        `sklearn.inspection`; this estimator implements nothing of its own.

        Examples
        --------
        >>> import numpy as np
        >>> from thiessen.estimators import AddiVortesRegressor
        >>> rng = np.random.default_rng(0)
        >>> X = rng.uniform(size=(60, 2))
        >>> y = 3.0 * (X[:, 0] - 0.4) ** 2 + 0.5 * X[:, 1]
        >>> model = AddiVortesRegressor(m=10, burn_in=20, draws=40, random_state=1)
        >>> model.fit(X, y).predict(X).shape
        (60,)

        The paper's cell-count prior is a keyword argument:

        >>> paper = AddiVortesRegressor(lambda_c=25, m=10, burn_in=20, draws=40)
    """

    def __init__(
        self,
        *,
        model: str = "gaussian",
        m_var: int = 40,
        m: int = 200,
        nu: float = 6.0,
        q: float = 0.85,
        k: float = 3.0,
        sigma_c: float = 0.8,
        omega: float | None = None,
        lambda_c: float = 5.0,
        burn_in: int = 200,
        draws: int = 1000,
        thinning: int = 1,
        prior_only: bool = False,
        metric: Sequence[MetricEntry] | None = None,
        categorical_features: str | Sequence[Any] | None = None,
        random_state: SeedLike = None,
    ) -> None:
        super().__init__(
            m=m,
            nu=nu,
            q=q,
            k=k,
            sigma_c=sigma_c,
            omega=omega,
            lambda_c=lambda_c,
            burn_in=burn_in,
            draws=draws,
            thinning=thinning,
            prior_only=prior_only,
            metric=metric,
            categorical_features=categorical_features,
            random_state=random_state,
        )
        self.model = model
        self.m_var = m_var

    def _model_name(self) -> str:
        if self.model not in {"gaussian", "heteroscedastic"}:
            if self.model == "probit":
                raise ValueError(
                    "model='probit' is a classifier; use AddiVortesClassifier"
                )
            raise ValueError(
                f"model must be 'gaussian' or 'heteroscedastic', got {self.model!r}"
            )
        return self.model

    def fit(self, X: Any, y: Any) -> AddiVortesRegressor:
        """Fit the model.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        y : array_like of shape (n_samples,)
            The response.

        Returns
        -------
        AddiVortesRegressor
            The fitted estimator.

        Raises
        ------
        ThiessenError
            For data the core rejects: missing or non-finite values, a
            constant response, a constant column, or fewer than two rows.
        """
        design, response = self._validate_fit(X, y)
        self._fit_core(design, response)
        return self

    def predict(
        self, X: Any, return_std: bool = False
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ...]:
        """Predict the mean function.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        return_std : bool, default=False
            Also return the posterior standard deviation of the mean function
            over the kept draws (`GaussianProcessRegressor` precedent).

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The posterior mean of f(x).
        y_std : ndarray of shape (n_samples,)
            Present when `return_std` is True.
        """
        design = self._validate_predict(X)
        if not return_std:
            return np.asarray(self._fitted.predict(design), dtype=np.float64)
        draws = self._fitted.predict_draws(design)
        return draws.mean(axis=0), draws.std(axis=0)

    def predict_interval(self, X: Any, level: float = 0.95) -> npt.NDArray[np.float64]:
        """Return the central posterior predictive interval.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        level : float, default=0.95
            Interval level in (0, 1).

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Lower and upper ends for a new observation.
        """
        design = self._validate_predict(X)
        return np.asarray(
            self._fitted.prediction_interval(design, float(level)), dtype=np.float64
        )


@_with_common
class AddiVortesClassifier(ClassifierMixin, _BaseAddiVortes):
    """AddiVortes binary classification.

        The probit model P(y = 1 | x) = Phi(c + f(x)) with Albert and Chib (1993)
        augmentation. Two classes only.

        Parameters
        ----------
        offset : float, default=None
            The offset c. `None` resolves to Phi^-1(ybar) at fit, the BART
            `binaryOffset` default.
    {common}

        Attributes
        ----------
        classes_ : ndarray of shape (2,)
            The class labels.
        n_features_in_ : int
            Number of features seen at fit.
        feature_names_in_ : ndarray of shape (n_features_in_,)
            Feature names seen at fit, when the input carried them.
        random_state_ : int
            The resolved seed.

        See Also
        --------
        AddiVortesRegressor : The Gaussian and heteroscedastic models.

        Examples
        --------
        >>> import numpy as np
        >>> from thiessen.estimators import AddiVortesClassifier
        >>> rng = np.random.default_rng(0)
        >>> X = rng.uniform(size=(60, 2))
        >>> y = (X[:, 0] > 0.5).astype(int)
        >>> model = AddiVortesClassifier(m=10, burn_in=20, draws=40, random_state=1)
        >>> model.fit(X, y).predict_proba(X).shape
        (60, 2)
    """

    def __init__(
        self,
        *,
        offset: float | None = None,
        m: int = 200,
        nu: float = 6.0,
        q: float = 0.85,
        k: float = 3.0,
        sigma_c: float = 0.8,
        omega: float | None = None,
        lambda_c: float = 5.0,
        burn_in: int = 200,
        draws: int = 1000,
        thinning: int = 1,
        prior_only: bool = False,
        metric: Sequence[MetricEntry] | None = None,
        categorical_features: str | Sequence[Any] | None = None,
        random_state: SeedLike = None,
    ) -> None:
        super().__init__(
            m=m,
            nu=nu,
            q=q,
            k=k,
            sigma_c=sigma_c,
            omega=omega,
            lambda_c=lambda_c,
            burn_in=burn_in,
            draws=draws,
            thinning=thinning,
            prior_only=prior_only,
            metric=metric,
            categorical_features=categorical_features,
            random_state=random_state,
        )
        self.offset = offset

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def _model_name(self) -> str:
        return "probit"

    def fit(self, X: Any, y: Any) -> AddiVortesClassifier:
        """Fit the model.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.
        y : array_like of shape (n_samples,)
            The class labels, two distinct values.

        Returns
        -------
        AddiVortesClassifier
            The fitted estimator.

        Raises
        ------
        ValueError
            If `y` does not hold exactly two classes.
        """
        design, labels = self._validate_fit(X, y)
        target = type_of_target(labels, input_name="y", raise_unknown=True)
        if target != "binary":
            raise ValueError(
                "Only binary classification is supported. The type of the "
                f"target is {target}."
            )
        self.classes_ = unique_labels(labels)
        if self.classes_.size < 2:
            raise ValueError(
                "AddiVortesClassifier needs two classes; y holds "
                f"{self.classes_.size} class."
            )
        indicator = (np.asarray(labels) == self.classes_[1]).astype(np.float64)
        self._fit_core(design, indicator)
        return self

    def predict_proba(self, X: Any) -> npt.NDArray[np.float64]:
        """Return the posterior mean class probabilities.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Columns in the order of `classes_`.
        """
        design = self._validate_predict(X)
        positive = np.asarray(self._fitted.predict(design), dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X: Any) -> npt.NDArray[Any]:
        """Return the class of highest posterior mean probability.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            The design.

        Returns
        -------
        ndarray of shape (n_samples,)
            Labels drawn from `classes_`.
        """
        indices = self.predict_proba(X).argmax(axis=1)
        return np.asarray(self.classes_)[indices]
