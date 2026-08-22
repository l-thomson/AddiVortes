"""The core's errors as the package's exception."""

from __future__ import annotations

import numpy as np
import pytest
from thiessen import GeometryParams, Model, TermParams, ThiessenError, _native

from .conftest import SMALL


def test_the_exception_is_a_value_error():
    assert issubclass(ThiessenError, ValueError)


def test_a_row_count_mismatch_is_rejected(gaussian_fixture):
    x, y = gaussian_fixture
    with pytest.raises(ThiessenError):
        Model(**SMALL).fit(x, y[:-1], random_state=1)


def test_a_non_finite_value_is_rejected(gaussian_fixture):
    x, y = gaussian_fixture
    x = x.copy()
    x[0, 0] = np.nan
    with pytest.raises(ThiessenError):
        Model(**SMALL).fit(x, y, random_state=1)


def test_a_constant_response_is_rejected(gaussian_fixture):
    x, _ = gaussian_fixture
    with pytest.raises(ThiessenError):
        Model(**SMALL).fit(x, np.ones(x.shape[0]), random_state=1)


def test_a_constant_column_is_rejected(gaussian_fixture):
    x, y = gaussian_fixture
    x = x.copy()
    x[:, 1] = 1.0
    with pytest.raises(ThiessenError):
        Model(**SMALL).fit(x, y, random_state=1)


def test_a_one_dimensional_design_is_rejected(gaussian_fixture):
    _, y = gaussian_fixture
    with pytest.raises(ValueError, match="two-dimensional"):
        Model(**SMALL).fit(np.arange(48.0), y, random_state=1)


def test_a_predict_column_mismatch_is_rejected(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1)
    with pytest.raises(ThiessenError):
        fitted.predict(x[:, :1])


def test_an_unknown_keyword_is_rejected():
    with pytest.raises(TypeError):
        Model(nonexistent=1.0)  # type: ignore[call-arg]


def test_the_core_rejects_an_unknown_configuration_field():
    with pytest.raises(ThiessenError, match="nonexistent"):
        _native.validate_config('{"nonexistent": 1.0}')


def test_an_unknown_outcome_name_is_rejected_by_the_core():
    with pytest.raises(ThiessenError, match="unknown variant `quantile`"):
        _native.validate_config('{"outcome": {"quantile": {}}}')


def test_an_unknown_group_setting_names_the_group():
    with pytest.raises(TypeError, match="TermParams"):
        TermParams(zeta=1.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="TermParams"):
        TermParams().set_params(zeta=1.0)


def test_a_group_of_the_wrong_type_is_rejected():
    with pytest.raises(TypeError, match="mean_params"):
        Model(mean_params={"tessellations": 8}).validate()  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="outcome"):
        Model(outcome="gaussian").validate()  # type: ignore[arg-type]


def test_a_non_integer_category_code_is_rejected():
    n = 30
    x = np.column_stack([np.linspace(0.0, 1.0, n), np.linspace(0.0, 3.0, n)])
    y = np.linspace(0.0, 1.0, n)
    geometry = GeometryParams(metric=["euclidean", "categorical"])
    with pytest.raises(ThiessenError):
        Model(
            mean_params=TermParams(tessellations=8, geometry=geometry),
            burn_in=10,
            draws=20,
        ).fit(x, y, random_state=1)


def test_an_invalid_probability_is_rejected(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1)
    with pytest.raises(ThiessenError):
        fitted.credible_interval(x, level=1.5)
    with pytest.raises(ThiessenError):
        fitted.predict_quantiles(x, [])
