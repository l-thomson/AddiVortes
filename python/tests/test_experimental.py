"""The exposure policy for items behind the core's `experimental` feature.

This release builds the core without the feature, so the package exposes the
published models only. A gated outcome has no constructor in this package,
and a configuration naming a gated field or variant fails to deserialise in
the core, which is what makes the policy hold without a change here when an
item is added or graduates.
"""

from __future__ import annotations

import json

import pytest
from thiessen import Model, TermParams, ThiessenError, _native, probit
from thiessen.estimators import AddiVortesClassifier
from thiessen.families import Probit

from .conftest import SMALL

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

#: Configuration fields that exist behind the feature only.
GATED_FIELDS = (
    ("mean_params", {"geometry": {"membership": "soft"}}),
    ("mean_params", {"geometry": {"precision": [1.0]}}),
    ("mean_params", {"structure": {"inclusion": "uniform"}}),
    ("mean_params", {"cell": {"basis": "linear"}}),
)


def test_the_extension_is_built_without_the_feature():
    assert _native.EXPERIMENTAL is False


def test_the_published_models_are_accepted():
    Model().validate()
    Model(outcome=probit()).validate()
    Model(variance_params=TermParams(tessellations=40)).validate()


@pytest.mark.parametrize("name", GATED)
def test_a_gated_outcome_fails_to_deserialise(name):
    with pytest.raises(ThiessenError):
        _native.validate_config(json.dumps({"outcome": {name: {}}}))


def test_no_constructor_exists_for_a_gated_outcome():
    from thiessen import families, params

    for module in (families, params):
        exposed = {name.lower() for name in dir(module)}
        assert not exposed & set(GATED)
    assert set(families.__all__) == {"Gaussian", "Probit", "gaussian", "probit"}


@pytest.mark.parametrize(("group", "fields"), GATED_FIELDS)
def test_a_gated_field_fails_to_deserialise(group, fields):
    with pytest.raises(ThiessenError):
        _native.validate_config(json.dumps({group: fields}))


def test_a_saved_model_naming_a_gated_option_fails_to_load(gaussian_fixture):
    x, y = gaussian_fixture
    payload = json.loads(Model(**SMALL).fit(x, y, random_state=1)._fitted.to_json())
    payload["config"]["outcome"] = {"soft": {}}

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
    assert isinstance(AddiVortesClassifier()._outcome(), Probit)
