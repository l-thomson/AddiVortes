"""The one seed rule of the package."""

from __future__ import annotations

import numpy as np
import pytest
from thiessen import Model
from thiessen._seed import _resolve_seed

SMALL = {"m": 8, "burn_in": 10, "draws": 20}


def test_an_integer_passes_through_unchanged():
    assert _resolve_seed(7) == 7
    assert _resolve_seed(np.int64(7)) == 7


def test_none_draws_fresh_entropy():
    assert _resolve_seed(None) != _resolve_seed(None)


def test_a_generator_supplies_one_draw():
    first = _resolve_seed(np.random.default_rng(0))
    second = _resolve_seed(np.random.default_rng(0))
    assert first == second
    assert 0 <= first < 1 << 64


def test_a_random_state_supplies_one_draw():
    first = _resolve_seed(np.random.RandomState(0))
    second = _resolve_seed(np.random.RandomState(0))
    assert first == second
    assert 0 <= first < 1 << 64


def test_an_out_of_range_integer_is_rejected():
    with pytest.raises(ValueError, match="2 \\*\\* 64"):
        _resolve_seed(1 << 64)
    with pytest.raises(ValueError, match="2 \\*\\* 64"):
        _resolve_seed(-1)


def test_an_unsupported_type_is_rejected():
    with pytest.raises(TypeError, match="random_state"):
        _resolve_seed("7")


def test_the_same_integer_gives_the_same_draws(gaussian_fixture):
    x, y = gaussian_fixture

    first = Model(**SMALL).fit(x, y, random_state=3)
    second = Model(**SMALL).fit(x, y, random_state=3)

    np.testing.assert_array_equal(first.predict_draws(x), second.predict_draws(x))
    assert first.random_state == second.random_state == 3


def test_different_seeds_give_different_draws(gaussian_fixture):
    x, y = gaussian_fixture

    first = Model(**SMALL).fit(x, y, random_state=3)
    second = Model(**SMALL).fit(x, y, random_state=4)

    assert not np.array_equal(first.predict_draws(x), second.predict_draws(x))


def test_the_resolved_seed_is_stored(gaussian_fixture):
    x, y = gaussian_fixture

    fitted = Model(**SMALL).fit(x, y, random_state=np.random.default_rng(0))

    assert fitted.random_state == _resolve_seed(np.random.default_rng(0))
