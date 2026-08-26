"""Several chains, pooled draws, and the convergence warning."""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest
from thiessen import Model
from thiessen._convergence import (
    _convergence_message,
    _monitored_rows,
)

from .conftest import SMALL


def fit_chains(fixture, n_chains=2, **params):
    """Fit with the short schedule, which never meets the thresholds."""
    x, y = fixture
    with pytest.warns(UserWarning):
        return Model(**SMALL, **params).fit(x, y, random_state=1, n_chains=n_chains)


def test_the_chains_pool_their_draws(gaussian_fixture):
    x, y = gaussian_fixture
    one = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    two = fit_chains(gaussian_fixture)

    assert one.n_chains == 1
    assert two.n_chains == 2
    assert two.n_draws == 2 * one.n_draws
    assert two.sigma().shape == (2 * one.n_draws,)


def test_the_first_chain_is_the_one_chain_fit(gaussian_fixture):
    x, y = gaussian_fixture
    one = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)

    two = fit_chains(gaussian_fixture)

    # Chain 0 is the seed itself, so the first chain repeats the one-chain fit.
    np.testing.assert_array_equal(two.sigma()[: one.n_draws], one.sigma())


def test_the_same_seed_reproduces_every_chain(gaussian_fixture):
    x, _ = gaussian_fixture

    first = fit_chains(gaussian_fixture, n_chains=3)
    second = fit_chains(gaussian_fixture, n_chains=3)

    np.testing.assert_array_equal(second.predict(x), first.predict(x))
    np.testing.assert_array_equal(second.sigma(), first.sigma())


def test_the_pooled_prediction_is_the_mean_of_the_pooled_draws(gaussian_fixture):
    x, _ = gaussian_fixture
    pooled = fit_chains(gaussian_fixture)

    draws = pooled.predict_draws(x[:5])

    assert draws.shape == (40, 5)
    np.testing.assert_allclose(pooled.predict(x[:5]), draws.mean(axis=0))


def test_the_inference_data_carries_the_chains(gaussian_fixture):
    pytest.importorskip("arviz")
    x, y = gaussian_fixture
    pooled = fit_chains(gaussian_fixture)

    posterior = pooled.to_inference_data(x, y)["posterior"].dataset

    assert posterior["mu"].shape == (2, 20, 48)
    assert posterior["sigma"].shape == (2, 20)


def test_a_short_run_warns_that_it_may_not_have_converged(gaussian_fixture):
    x, y = gaussian_fixture

    with pytest.warns(UserWarning, match="may not have converged"):
        Model(**SMALL).fit(x, y, random_state=1, n_chains=2)


def test_one_chain_does_not_warn(gaussian_fixture):
    x, y = gaussian_fixture

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Model(**SMALL).fit(x, y, random_state=1, n_chains=1)


def test_a_pooled_fit_pickles_with_its_chain_count(gaussian_fixture):
    x, _ = gaussian_fixture
    pooled = fit_chains(gaussian_fixture)

    restored = pickle.loads(pickle.dumps(pooled))

    assert restored.n_chains == 2
    np.testing.assert_array_equal(restored.predict(x), pooled.predict(x))


@pytest.mark.parametrize("n_chains", [0, -1, 1.5])
def test_the_chain_count_must_be_a_positive_integer(gaussian_fixture, n_chains):
    x, y = gaussian_fixture

    with pytest.raises(ValueError, match="positive integer"):
        Model(**SMALL).fit(x, y, random_state=1, n_chains=n_chains)


def test_the_message_states_both_thresholds():
    bad = {"rhat": 1.2, "ess_bulk": 100.0, "ess_tail": 150.0}
    good = {"rhat": 1.0, "ess_bulk": 500.0, "ess_tail": 500.0}

    message = _convergence_message(bad)

    assert message is not None
    assert "R-hat 1.200" in message
    assert "sample size 100" in message
    assert _convergence_message(good) is None


def test_the_monitored_rows_are_a_subsample():
    np.testing.assert_array_equal(_monitored_rows(5, 20), np.arange(5))
    assert _monitored_rows(1000, 20).shape == (20,)
    assert _monitored_rows(1000, 20)[0] == 0
    assert _monitored_rows(1000, 20)[-1] == 999


def test_threaded_chains_draw_what_the_chains_run_in_turn_draw(gaussian_fixture):
    x, y = gaussian_fixture
    serial = fit_chains(gaussian_fixture, n_chains=3)

    for n_threads in (2, 3, 8):
        with pytest.warns(UserWarning):
            threaded = Model(**SMALL).fit(
                x, y, random_state=1, n_chains=3, n_threads=n_threads
            )

        assert threaded.n_threads == n_threads
        np.testing.assert_array_equal(threaded.sigma(), serial.sigma())
        np.testing.assert_array_equal(
            threaded.predict_draws(x), serial.predict_draws(x)
        )
        np.testing.assert_array_equal(
            threaded.prediction_interval(x), serial.prediction_interval(x)
        )
    assert serial.n_threads == 1


def test_the_thread_count_survives_a_pickle(gaussian_fixture):
    x, y = gaussian_fixture
    with pytest.warns(UserWarning):
        threaded = Model(**SMALL).fit(x, y, random_state=1, n_chains=2, n_threads=2)

    loaded = pickle.loads(pickle.dumps(threaded))

    assert loaded.n_threads == 2
    np.testing.assert_array_equal(loaded.predict(x), threaded.predict(x))


@pytest.mark.parametrize("n_threads", [0, -1, 1.5, "two"])
def test_the_thread_count_must_be_a_positive_integer(gaussian_fixture, n_threads):
    x, y = gaussian_fixture
    with pytest.raises((ValueError, TypeError)):
        Model(**SMALL).fit(x, y, random_state=1, n_threads=n_threads, n_chains=1)


def test_the_thread_count_can_be_set_on_a_fitted_model(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1, n_chains=1)
    mean = fitted.predict(x)
    interval = fitted.prediction_interval(x)

    fitted.n_threads = 2

    assert fitted.n_threads == 2
    np.testing.assert_array_equal(fitted.predict(x), mean)
    np.testing.assert_array_equal(fitted.prediction_interval(x), interval)
    with pytest.raises(ValueError):
        fitted.n_threads = 0


def test_a_fit_defaults_to_four_chains_on_one_thread(gaussian_fixture):
    x, y = gaussian_fixture

    with pytest.warns(UserWarning):
        fitted = Model(**SMALL).fit(x, y, random_state=1)

    assert fitted.n_chains == 4
    assert fitted.n_threads == 1
    assert fitted.n_draws == 4 * SMALL["draws"]
