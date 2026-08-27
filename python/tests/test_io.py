"""Saving and loading a fitted model, and the fit-time warnings."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from thiessen import FittedModel, Model, ThiessenError
from thiessen.estimators import AddiVortesRegressor

from .conftest import SMALL


def _wide():
    rng = np.random.default_rng(0)
    return rng.uniform(size=(4, 6)), rng.uniform(size=4)


def test_save_and_load_round_trip(tmp_path, gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)
    path = tmp_path / "fit.json"

    fitted.save(path)
    loaded = FittedModel.load(path)

    np.testing.assert_array_equal(loaded.predict(x), fitted.predict(x))
    assert loaded.model == fitted.model
    assert loaded.n_draws == fitted.n_draws


def test_save_accepts_a_string_path(tmp_path, gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)
    path = str(tmp_path / "fit.json")

    fitted.save(path)

    assert FittedModel.load(path).n_draws == 20


def test_load_reports_the_seed_it_is_given(tmp_path, gaussian_fixture):
    x, y = gaussian_fixture
    path = tmp_path / "fit.json"
    Model(**SMALL).fit(x, y, random_state=1, n_chains=1).save(path)

    assert FittedModel.load(path, random_state=7).random_state == 7


def test_saving_to_a_missing_directory_raises_os_error(tmp_path, gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    with pytest.raises(OSError):
        fitted.save(tmp_path / "absent" / "fit.json")


def test_loading_a_missing_file_raises_os_error(tmp_path):
    with pytest.raises(OSError):
        FittedModel.load(tmp_path / "absent.json")


def test_loading_a_file_that_is_not_a_model(tmp_path):
    path = tmp_path / "fit.json"
    path.write_text("{}")

    with pytest.raises(ThiessenError):
        FittedModel.load(path)


def test_loading_a_directory_raises_os_error(tmp_path):
    with pytest.raises(OSError):
        FittedModel.load(tmp_path)


def test_a_path_like_object_is_accepted(tmp_path, gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)
    path = Path(tmp_path) / "fit.json"

    fitted.save(path)

    assert FittedModel.load(Path(path)).n_draws == 20


def test_fit_warns_when_features_outnumber_observations():
    x, y = _wide()

    with pytest.warns(UserWarning, match="more features"):
        Model(**SMALL).fit(x, y, random_state=1, n_chains=1)


def test_the_estimators_warn_at_fit():
    x, y = _wide()

    with pytest.warns(UserWarning, match="more features"):
        AddiVortesRegressor(random_state=1, **SMALL, n_chains=1).fit(x, y)


def test_a_clean_fit_warns_of_nothing(gaussian_fixture):
    x, y = gaussian_fixture

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Model(**SMALL).fit(x, y, random_state=1, n_chains=1)


def test_the_warnings_stay_on_the_fitted_object():
    x, y = _wide()

    with pytest.warns(UserWarning):
        fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    assert any("more features" in message for message in fitted.warnings)
