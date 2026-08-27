"""The exposure policy for items behind the core's `experimental` feature.

A constructor exists in every build and the core reports a gated item with
`RequiresFeatureError`, so the tests read the families from the core and
hold no list of their own: an item added or graduated needs no change
here. A wheel is built without the feature; a build with it is made from
source with ``--features experimental``.
"""

from __future__ import annotations

import json

import pytest
from thiessen import Model, RequiresFeatureError, TermParams, ThiessenError, _native
from thiessen import families as families_module
from thiessen.estimators import AddiVortesClassifier
from thiessen.families import Probit
from thiessen.params import _TAGGED_ENTRIES

from .conftest import SMALL

#: The metric variants of `crates/thiessen/src/geometry.rs` with no
#: required field. A unit variant is stored as a bare string and a struct
#: variant as a map, which the surface tags for the caller.
METRICS_WITHOUT_REQUIRED_FIELDS = {
    "euclidean": "unit",
    "categorical": "unit",
    "mahalanobis": "unit",
    "manhattan": "struct",
    "cosine": "struct",
}


def core_families() -> set[str]:
    """The outcome families the core in use carries, by their stored names."""
    catalogue = json.loads(_native.outcome_defaults())
    return {next(iter(family)) for family in catalogue}


def surface_families() -> set[str]:
    """The families this package constructs, by their stored names."""
    return {name for name in families_module.__all__ if name.islower()}


def outcome_of(kind: str):
    """The family's constructor, called at its defaults."""
    return getattr(families_module, kind)()


def test_every_family_the_core_carries_has_a_constructor():
    assert core_families() <= surface_families()


@pytest.mark.skipif(not _native.EXPERIMENTAL, reason="built without the feature")
def test_a_build_with_the_feature_turns_no_family_away():
    assert surface_families() == core_families()
    for kind in surface_families():
        # A family may still be rejected on its own terms, the tobit
        # outcome needing a censoring limit; none is rejected for the
        # feature.
        try:
            Model(outcome=outcome_of(kind)).validate()
        except RequiresFeatureError:  # pragma: no cover - the failure itself
            pytest.fail(f"{kind} was turned away for the feature")
        except ThiessenError:
            pass


@pytest.mark.skipif(_native.EXPERIMENTAL, reason="built with the feature")
def test_a_build_without_the_feature_reports_each_gated_family():
    gated = surface_families() - core_families()

    assert gated
    for kind in gated:
        with pytest.raises(RequiresFeatureError):
            Model(outcome=outcome_of(kind)).validate()


def test_the_published_models_are_accepted():
    Model().validate()
    Model(outcome=families_module.probit()).validate()
    Model(variance_params=TermParams(tessellations=40)).validate()


@pytest.mark.skipif(not _native.EXPERIMENTAL, reason="built without the feature")
def test_a_degrees_of_freedom_grid_crosses_as_an_array():
    Model(outcome=families_module.student_t(df=[3.0, 6.0, 12.0])).validate()


@pytest.mark.skipif(_native.EXPERIMENTAL, reason="built with the feature")
def test_a_gated_component_option_reports_the_feature():
    config = '{"mean_params": {"geometry": {"membership": {"soft": {}}}}}'

    with pytest.raises(RequiresFeatureError):
        _native.validate_config(config)


@pytest.mark.parametrize(
    "config",
    [
        {"mean_params": {"geometry": {"membership": "hard"}}},
        {"mean_params": {"structure": {"inclusion": "uniform"}}},
        {"mean_params": {"cell": {"basis": "constant"}}},
    ],
)
def test_the_published_default_of_a_gated_field_is_accepted(config):
    _native.validate_config(json.dumps(config))


def test_an_invalid_configuration_keeps_the_plain_exception():
    config = {"mean_params": {"geometry": {"sigma_c": -1.0}}}

    with pytest.raises(ThiessenError) as raised:
        _native.validate_config(json.dumps(config))
    assert not isinstance(raised.value, RequiresFeatureError)


def test_an_unknown_field_is_a_deserialisation_error():
    with pytest.raises(ThiessenError, match="nonexistent"):
        _native.validate_config('{"nonexistent": true}')


@pytest.mark.skipif(_native.EXPERIMENTAL, reason="built with the feature")
def test_a_saved_model_naming_a_gated_family_reports_the_feature(gaussian_fixture):
    x, y = gaussian_fixture
    payload = json.loads(Model(**SMALL).fit(x, y, random_state=1)._fitted.to_json())
    payload["config"]["outcome"] = {"laplace": {}}

    with pytest.raises(RequiresFeatureError):
        _native.fitted_from_json(json.dumps(payload))


def test_a_saved_model_naming_an_unknown_family_fails_to_load(gaussian_fixture):
    x, y = gaussian_fixture
    payload = json.loads(Model(**SMALL).fit(x, y, random_state=1)._fitted.to_json())
    payload["config"]["outcome"] = {"robust": {}}

    with pytest.raises(ThiessenError):
        _native.fitted_from_json(json.dumps(payload))


def metric_config(entry) -> str:
    return json.dumps({"mean_params": {"geometry": {"metric": [entry]}}})


def test_the_tagged_entries_are_the_struct_variants_without_required_fields():
    structs = {
        name
        for name, form in METRICS_WITHOUT_REQUIRED_FIELDS.items()
        if form == "struct"
    }
    assert set(_TAGGED_ENTRIES) == structs
    for name, form in METRICS_WITHOUT_REQUIRED_FIELDS.items():
        stored = name if form == "unit" else {name: {}}
        other = {name: {}} if form == "unit" else name
        try:
            _native.validate_config(metric_config(stored))
        except RequiresFeatureError:
            assert not _native.EXPERIMENTAL
        with pytest.raises(ThiessenError) as refused:
            _native.validate_config(metric_config(other))
        assert not isinstance(refused.value, RequiresFeatureError)


def test_the_classifier_is_always_probit():
    assert isinstance(AddiVortesClassifier()._outcome(), Probit)
