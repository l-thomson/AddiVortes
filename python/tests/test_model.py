"""The configuration and the fitted model."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from thiessen import CORE_VERSION, FittedModel, Model, ThiessenError

from .conftest import SMALL


def test_gaussian_fit_predicts_at_every_row(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1)

    assert isinstance(fitted, FittedModel)
    assert fitted.model == "gaussian"
    assert fitted.n_draws == 20
    assert fitted.predict(x).shape == (48,)
    assert fitted.predict_draws(x).shape == (20, 48)
    assert fitted.in_sample_rmse < y.std()


def test_defaults_come_from_the_core(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1)

    config = fitted.config
    assert config["model"] == "gaussian"
    assert config["lambda_c"] == 5.0
    assert config["nu"] == 6.0
    assert config["q"] == 0.85
    assert config["k"] == 3.0
    # omega resolves at fit to min(3, p).
    assert config["omega"] == 2.0


def test_posterior_accessors_have_one_value_per_draw(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1)

    assert fitted.sigma().shape == (20,)
    assert fitted.cell_counts().shape == (20,)
    assert fitted.dimension_counts().shape == (20,)
    proportions = fitted.variable_inclusion_proportions()
    assert proportions.shape == (2,)
    assert proportions.sum() == pytest.approx(1.0)


def test_intervals_and_quantiles(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1)

    quantiles = fitted.predict_quantiles(x, [0.1, 0.5, 0.9])
    assert quantiles.shape == (48, 3)
    assert np.all(np.diff(quantiles, axis=1) >= 0.0)

    credible = fitted.credible_interval(x, level=0.9)
    assert credible.shape == (48, 2)
    assert np.all(credible[:, 0] <= credible[:, 1])

    predictive = fitted.prediction_interval(x, level=0.9)
    assert np.all(predictive[:, 0] <= credible[:, 0])
    assert np.all(predictive[:, 1] >= credible[:, 1])


def test_log_likelihood_is_draw_major(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1)

    assert fitted.log_likelihood(x, y).shape == (20, 48)


def test_probit_predicts_probabilities(probit_fixture):
    x, y = probit_fixture
    fitted = Model(model="probit", **SMALL).fit(x, y, random_state=1)

    probabilities = fitted.predict(x)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert fitted.sigma().shape == (0,)
    with pytest.raises(ThiessenError, match="probit"):
        fitted.predict_variance(x)
    with pytest.raises(ThiessenError, match="probit"):
        fitted.prediction_interval(x)


def test_heteroscedastic_variance_varies_by_row(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(model="heteroscedastic", m_var=5, **SMALL).fit(x, y, random_state=1)

    variance = fitted.predict_variance(x)
    assert variance.shape == (20, 48)
    assert np.all(variance > 0.0)
    assert np.ptp(variance[0]) > 0.0


def test_gaussian_variance_is_constant_across_rows(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1)

    variance = fitted.predict_variance(x)
    assert np.ptp(variance[0]) == 0.0


def test_prior_only_ignores_the_response(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(prior_only=True, **SMALL).fit(x, y, random_state=1)

    assert fitted.in_sample_rmse > 0.0
    assert fitted.config["prior_only"] is True


def test_pickle_round_trip_preserves_predictions(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1)

    restored = pickle.loads(pickle.dumps(fitted))

    assert restored.random_state == fitted.random_state
    assert restored.model == fitted.model
    np.testing.assert_array_equal(restored.predict(x), fitted.predict(x))


def test_warnings_report_more_features_than_observations():
    rng = np.random.default_rng(0)
    x = rng.uniform(size=(4, 6))
    y = rng.uniform(size=4)

    fitted = Model(**SMALL).fit(x, y, random_state=1)

    assert any("more features" in warning for warning in fitted.warnings)


def test_metric_accepts_a_spherical_column():
    n = 30
    latitude = np.linspace(-1.0, 1.0, n)
    longitude = np.linspace(-3.0, 3.0, n)
    x = np.column_stack([latitude, longitude])
    y = np.sin(latitude) + 0.1 * longitude

    fitted = Model(
        metric=[{"spherical": {"sphere": 0}}, {"spherical": {"sphere": 0}}],
        **SMALL,
    ).fit(x, y, random_state=1)

    assert fitted.predict(x).shape == (n,)


def test_metric_accepts_a_categorical_column():
    n = 40
    codes = np.array([i % 4 for i in range(n)], dtype=np.float64)
    continuous = np.linspace(0.0, 1.0, n)
    x = np.column_stack([continuous, codes])
    y = continuous + codes

    fitted = Model(metric=["euclidean", "categorical"], **SMALL).fit(
        x, y, random_state=1
    )

    assert fitted.predict(x).shape == (n,)


def test_repr_shows_only_the_set_fields():
    assert repr(Model(m=10)) == "Model(m=10)"


def test_validate_rejects_a_bad_hyperparameter():
    with pytest.raises(ThiessenError, match="m"):
        Model(m=0).validate()


def test_core_version_is_reported():
    assert CORE_VERSION.count(".") == 2
