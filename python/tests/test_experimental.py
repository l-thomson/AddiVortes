"""The exposure policy for items behind the core's `experimental` feature.

This release builds the core without the feature, so the package exposes the
published models only. The package keeps no list of model names: every name is
validated by the core, which is what makes the policy hold without a change
here when an item is added or graduates.
"""

from __future__ import annotations

import json

import pytest
from thiessen import Model, ThiessenError, _native
from thiessen.estimators import AddiVortesClassifier, AddiVortesRegressor

SMALL = {"m": 8, "burn_in": 10, "draws": 20}

PUBLISHED = ("gaussian", "probit", "heteroscedastic")

#: Names reserved for items behind the feature; none is accepted here.
GATED = (
    "soft",
    "robust_t",
    "dart",
    "minkowski",
    "manhattan",
    "mahalanobis",
    "gower",
    "cosine",
    "weighted",
    "composite",
)


def test_the_extension_is_built_without_the_feature():
    assert _native.EXPERIMENTAL is False


def test_the_published_models_are_accepted():
    for name in PUBLISHED:
        Model(model=name).validate()


@pytest.mark.parametrize("name", GATED)
def test_a_gated_name_is_rejected(name):
    with pytest.raises(ThiessenError):
        Model(model=name).validate()


@pytest.mark.parametrize("name", GATED)
def test_the_estimators_reject_a_gated_name(name, gaussian_fixture):
    x, y = gaussian_fixture
    with pytest.raises(ValueError):
        AddiVortesRegressor(model=name, **SMALL).fit(x, y)


def test_the_estimators_hold_no_model_list():
    """The estimator passes the name through; the core rejects it."""
    assert AddiVortesRegressor(model="soft")._model_name() == "soft"
    with pytest.raises(ThiessenError, match="unknown model"):
        Model(model="soft").validate()


def test_a_saved_model_naming_a_gated_option_fails_to_load(gaussian_fixture):
    x, y = gaussian_fixture
    payload = json.loads(Model(**SMALL).fit(x, y, random_state=1)._fitted.to_json())
    payload["config"]["model"] = "soft"

    with pytest.raises(ThiessenError):
        _native.fitted_from_json(json.dumps(payload))


def test_a_configuration_naming_a_gated_field_fails_to_load():
    with pytest.raises(ThiessenError, match="nonexistent"):
        _native.validate_config('{"nonexistent": true}')


def test_the_extension_exposes_no_gated_names():
    exposed = {name.lower() for name in dir(_native)}
    assert not exposed & set(GATED)


def test_the_package_exposes_no_gated_names():
    import thiessen

    exposed = {name.lower() for name in dir(thiessen)}
    assert not exposed & set(GATED)
    assert "experimental" not in vars(thiessen)


def test_the_classifier_is_always_probit():
    assert AddiVortesClassifier()._model_name() == "probit"
