"""The sampler API against the fitting entry point."""

from __future__ import annotations

import numpy as np
import pytest
from thiessen import Model, TermParams, ThiessenError, probit
from thiessen.sampler import Sampler

from .conftest import SMALL


def _driven(x, y, **kwargs):
    """Drive the SMALL schedule by hand: 10 burn-in, 20 kept, thinning 1."""
    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), **kwargs)
    sampler.step(10)
    for _ in range(20):
        sampler.step(1)
        sampler.keep()
    return sampler


def test_a_driven_fit_matches_fit_bit_for_bit(gaussian_fixture):
    x, y = gaussian_fixture

    through_fit = Model(**SMALL).fit(x, y, random_state=1)
    through_sampler = _driven(x, y, random_state=1).finish()

    np.testing.assert_array_equal(
        through_sampler.predict_draws(x), through_fit.predict_draws(x)
    )
    np.testing.assert_array_equal(through_sampler.sigma(), through_fit.sigma())


def test_thinning_is_the_caller_loop(gaussian_fixture):
    x, y = gaussian_fixture
    thinned = Model(thinning=3, **SMALL).fit(x, y, random_state=1)

    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), random_state=1)
    sampler.step(10)
    for _ in range(20):
        sampler.step(3)
        sampler.keep()
    driven = sampler.finish()

    np.testing.assert_array_equal(driven.predict_draws(x), thinned.predict_draws(x))


def test_finish_returns_the_ordinary_fitted_model(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = _driven(x, y, random_state=1).finish()

    assert fitted.model == "gaussian"
    assert fitted.n_draws == 20
    assert fitted.predict(x).shape == (48,)
    assert fitted.config["mean_params"]["tessellations"] == 8


def test_n_kept_counts_the_kept_draws(gaussian_fixture):
    x, y = gaussian_fixture
    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), random_state=1)

    assert sampler.n_kept == 0
    sampler.step(2)
    sampler.keep()
    assert sampler.n_kept == 1


def test_fitted_values_and_noise_variances_have_one_value_per_row(gaussian_fixture):
    x, y = gaussian_fixture
    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), random_state=1)
    sampler.step(2)

    assert sampler.fitted_values().shape == (48,)
    variances = sampler.noise_variances()
    assert variances.shape == (48,)
    assert np.all(variances > 0.0)


def test_set_response_conditions_the_next_sweep(gaussian_fixture):
    x, y = gaussian_fixture
    unchanged = _driven(x, y, random_state=1).finish()

    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), random_state=1)
    sampler.step(10)
    sampler.set_response(y + 0.5 * np.sin(6.0 * x[:, 0]))
    for _ in range(20):
        sampler.step(1)
        sampler.keep()
    swapped = sampler.finish()

    assert not np.array_equal(swapped.predict_draws(x), unchanged.predict_draws(x))


def test_a_response_outside_the_training_range_is_legitimate(gaussian_fixture):
    x, y = gaussian_fixture
    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), random_state=1)
    sampler.step(2)

    sampler.set_response(y + 100.0)
    sampler.step(2)

    assert sampler.fitted_values().shape == (48,)


def test_set_response_rejections_keep_their_reason(gaussian_fixture):
    x, y = gaussian_fixture
    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), random_state=1)

    with pytest.raises(ThiessenError):
        sampler.set_response(y[:-1])
    bad = y.copy()
    bad[0] = np.nan
    with pytest.raises(ThiessenError):
        sampler.set_response(bad)


def test_probit_labels_are_validated(probit_fixture):
    x, y = probit_fixture
    sampler = Sampler(
        x, y, outcome=probit(), mean_params=TermParams(tessellations=8), random_state=1
    )
    sampler.step(2)

    with pytest.raises(ThiessenError):
        sampler.set_response(y + 0.5)


def test_finish_without_a_kept_draw_is_rejected(gaussian_fixture):
    x, y = gaussian_fixture
    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), random_state=1)
    sampler.step(2)

    with pytest.raises(ThiessenError, match="no draws were kept"):
        sampler.finish()


def test_every_call_after_finish_is_rejected(gaussian_fixture):
    x, y = gaussian_fixture
    sampler = _driven(x, y, random_state=1)
    sampler.finish()

    for call in (
        lambda: sampler.step(1),
        sampler.keep,
        sampler.fitted_values,
        sampler.noise_variances,
        sampler.finish,
        lambda: sampler.set_response(y),
    ):
        with pytest.raises(ThiessenError, match="finished"):
            call()
    assert "finished" in repr(sampler)


def test_a_negative_step_count_is_rejected(gaussian_fixture):
    x, y = gaussian_fixture
    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), random_state=1)

    with pytest.raises(ValueError, match="whole number"):
        sampler.step(-1)


def test_the_resolved_configuration_is_reported(gaussian_fixture):
    x, y = gaussian_fixture
    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), random_state=1)

    config = sampler.config
    assert config["mean_params"]["tessellations"] == 8
    assert config["mean_params"]["structure"]["omega"] == 2.0


def test_the_data_contract_applies_at_construction(gaussian_fixture):
    x, y = gaussian_fixture
    with pytest.raises(ThiessenError):
        Sampler(x, y[:-1], random_state=1)
