"""The response shapes and the outcome family each selects."""

from __future__ import annotations

import numpy as np
import pytest
from thiessen import Model, aft, gaussian, probit
from thiessen._response import _as_response, _resolve_outcome
from thiessen.families import Aft, Gaussian, IntervalCensored, Ordinal, Probit
from thiessen.sampler import Sampler

from .conftest import SMALL

pd = pytest.importorskip("pandas")


def survival(n):
    return np.array(
        [(i % 2 == 0, 1.0 + i) for i in range(n)],
        dtype=[("event", bool), ("time", float)],
    )


def test_a_numeric_array_is_a_plain_response():
    response = _as_response([1.0, 2.0, 3.0])

    assert (response.kind, response.shape, response.n) == ("plain", "numeric", 3)
    assert isinstance(_resolve_outcome(None, response), Gaussian)


def test_a_single_column_is_flattened():
    response = _as_response(np.ones((4, 1)))

    assert response.shape == "numeric"
    assert response.y.shape == (4,)


def test_a_boolean_array_selects_the_probit_family():
    response = _as_response(np.array([True, False, True]))

    assert response.shape == "binary"
    np.testing.assert_array_equal(response.y, [1.0, 0.0, 1.0])
    assert isinstance(_resolve_outcome(None, response), Probit)


def test_a_two_category_categorical_selects_the_probit_family():
    response = _as_response(pd.Categorical(["b", "a", "b"]))

    assert response.shape == "binary"
    assert response.categories == ("a", "b")
    np.testing.assert_array_equal(response.y, [1.0, 0.0, 1.0])


def test_an_ordered_categorical_selects_the_ordinal_family():
    y = pd.Series(pd.Categorical(["lo", "hi", "mid"], ["lo", "mid", "hi"], True))

    response = _as_response(y)

    assert response.shape == "ordered"
    np.testing.assert_array_equal(response.y, [0.0, 2.0, 1.0])
    outcome = _resolve_outcome(None, response)
    assert isinstance(outcome, Ordinal)
    assert outcome.categories == 3


def test_a_survival_array_selects_the_aft_family():
    response = _as_response(survival(4))

    assert (response.kind, response.shape) == ("aft", "right")
    np.testing.assert_array_equal(response.events, [True, False, True, False])
    np.testing.assert_array_equal(response.times, [1.0, 2.0, 3.0, 4.0])
    assert isinstance(_resolve_outcome(None, response), Aft)


def test_a_two_column_array_selects_the_interval_censored_family():
    response = _as_response([[0.0, 1.0], [-np.inf, 2.0], [3.0, 3.0]])

    assert (response.kind, response.shape) == ("interval_censored", "interval")
    np.testing.assert_array_equal(response.lower, [0.0, -np.inf, 3.0])
    assert isinstance(_resolve_outcome(None, response), IntervalCensored)


def test_a_missing_category_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        _as_response(pd.Categorical(["a", None, "b"]))


def test_three_unordered_categories_are_rejected():
    with pytest.raises(ValueError, match="two categories, or be ordered"):
        _as_response(pd.Categorical(["a", "b", "c"]))


def test_a_survival_array_needs_the_event_then_the_time():
    reversed_fields = np.array([(1.0, True)], dtype=[("time", float), ("event", bool)])

    with pytest.raises(ValueError, match="boolean event indicator then"):
        _as_response(reversed_fields)


@pytest.mark.parametrize(
    ("outcome", "y", "named"),
    [
        (probit(), survival(4), "probit"),
        (gaussian(), np.array([True, False, True, True]), "gaussian"),
        (aft(), np.arange(4.0), "aft"),
    ],
)
def test_a_named_family_the_response_does_not_fit_is_an_error(outcome, y, named):
    with pytest.raises(ValueError, match=f"outcome names the {named} family"):
        _resolve_outcome(outcome, _as_response(y))


def test_a_boolean_response_fits_the_probit_family_unnamed(probit_fixture):
    x, y = probit_fixture

    fitted = Model(**SMALL).fit(x, y.astype(bool), random_state=1, n_chains=1)

    assert fitted.model == "probit"
    assert fitted.predict(x).shape == (48,)
    np.testing.assert_array_equal(
        fitted.log_likelihood(x, y.astype(bool)), fitted.log_likelihood(x, y)
    )


def test_labels_in_zero_and_one_still_fit_the_named_probit_family(probit_fixture):
    x, y = probit_fixture

    named = Model(outcome=probit(), **SMALL).fit(x, y, random_state=1, n_chains=1)
    unnamed = Model(**SMALL).fit(x, y.astype(bool), random_state=1, n_chains=1)

    np.testing.assert_array_equal(named.predict_draws(x), unnamed.predict_draws(x))


def test_a_categorical_response_fits_the_probit_family(probit_fixture):
    x, y = probit_fixture
    labels = pd.Categorical(np.where(y > 0.5, "yes", "no"), ["no", "yes"])

    fitted = Model(**SMALL).fit(x, labels, random_state=1, n_chains=1)

    assert fitted.model == "probit"


def test_the_fitted_family_checks_the_response_shape(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    with pytest.raises(ValueError, match="selects the aft family"):
        fitted.log_likelihood(x, survival(48))
    with pytest.raises(ValueError, match="selects the aft family"):
        fitted.to_inference_data(x, survival(48))


def test_the_sampler_takes_the_response_shapes(probit_fixture):
    x, y = probit_fixture

    sampler = Sampler(x, y.astype(bool), random_state=1)

    assert "probit" in sampler.config["outcome"]
    with pytest.raises(ValueError, match="selects the aft family"):
        sampler.set_response(survival(48))
