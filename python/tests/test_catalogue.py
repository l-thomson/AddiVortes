"""One fit per row of docs/experimental.md through `Model`.

In a build with the feature each row fits, predicts, evaluates its
log-likelihood, round-trips through `save`/`load` and pickle, and converts
to an arviz `DataTree`; in a build without it each raises
`RequiresFeatureError`. The design has two columns, so a metric names two.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from thiessen import (
    CellParams,
    FittedModel,
    GeometryParams,
    Model,
    RequiresFeatureError,
    StructureParams,
    TermParams,
    _native,
    dart_inclusion,
    laplace,
    ordinal,
    soft_membership,
    student_t,
    tobit,
    weighted_inclusion,
)
from thiessen.sampler import Sampler

from .conftest import SMALL, _fixture, survival

try:
    import pandas as pd
except ImportError:  # pragma: no cover - depends on the install
    pd = None

experimental = pytest.mark.skipif(
    not _native.EXPERIMENTAL, reason="built without the feature"
)
default_build = pytest.mark.skipif(
    _native.EXPERIMENTAL, reason="built with the feature"
)
needs_pandas = pytest.mark.skipif(pd is None, reason="needs pandas")


def geometry(**kwargs):
    return TermParams(tessellations=8, geometry=GeometryParams(**kwargs))


def structure(inclusion):
    return TermParams(tessellations=8, structure=StructureParams(inclusion=inclusion))


def small(**kwargs):
    return Model(burn_in=10, draws=20, **kwargs)


def minkowski(p, group=0):
    return {"minkowski": {"p": p, "group": group}}


def catalogue_rows():
    """One `(model, y)` pair per row of the table, keyed by item."""
    x, y = _fixture()
    n = len(y)
    index = np.arange(n)
    # The ordinal row is an ordered `Categorical`; without pandas it is
    # skipped by `items`, not dropped.
    ordered = (
        None
        if pd is None
        else pd.Categorical(
            np.array(["lo", "mid", "hi"])[index % 3],
            categories=["lo", "mid", "hi"],
            ordered=True,
        )
    )
    times = np.exp(y)
    events = index % 3 != 2
    lower = np.where(index % 7 == 1, -np.inf, y - 0.1)
    upper = np.where(index % 7 == 4, np.inf, y + 0.1)
    floor = float(np.quantile(y, 0.2))
    gower = {"gower": {"kind": "numeric"}}
    return {
        "minkowski": (
            small(mean_params=geometry(metric=[minkowski(3.0), minkowski(3.0)])),
            y,
        ),
        "manhattan": (small(mean_params=geometry(metric=["manhattan"] * 2)), y),
        "cosine": (small(mean_params=geometry(metric=["cosine"] * 2)), y),
        "gower": (small(mean_params=geometry(metric=[gower, gower])), y),
        "mahalanobis": (
            small(
                mean_params=geometry(metric=["mahalanobis"] * 2, precision=np.eye(2))
            ),
            y,
        ),
        "composite": (
            small(
                mean_params=geometry(
                    metric=[minkowski(1.5, group=0), minkowski(3.0, group=1)]
                )
            ),
            y,
        ),
        "weighted": (
            small(mean_params=structure(weighted_inclusion([2.0, 1.0]))),
            y,
        ),
        "dart": (small(mean_params=structure(dart_inclusion())), y),
        "linear": (
            small(mean_params=TermParams(tessellations=8, cell=CellParams("linear"))),
            y,
        ),
        "soft": (small(mean_params=geometry(membership=soft_membership())), y),
        "tobit": (Model(outcome=tobit(lower=floor), **SMALL), np.maximum(y, floor)),
        "aft": (Model(**SMALL), survival(events, times)),
        "interval_censored": (Model(**SMALL), np.column_stack([lower, upper])),
        "ordinal": (Model(**SMALL), ordered),
        "student_t": (Model(outcome=student_t(df=4.0), **SMALL), y),
        "student_t_grid": (Model(outcome=student_t(df=[3.0, 6.0, 12.0]), **SMALL), y),
        "laplace": (Model(outcome=laplace(), **SMALL), y),
    }


ROWS = catalogue_rows()


def items():
    """The row keys, the ordinal row marked for its pandas dependency."""
    return [
        pytest.param(item, marks=needs_pandas) if item == "ordinal" else item
        for item in sorted(ROWS)
    ]


@experimental
@pytest.mark.parametrize("item", items())
def test_every_row_fits_predicts_and_round_trips(item, tmp_path):
    x, _ = _fixture()
    model, y = ROWS[item]

    fitted = model.fit(x, y, random_state=1)

    assert fitted.n_draws == 20
    assert fitted.predict(x).shape == (48,)
    assert fitted.log_likelihood(x, y).shape == (20, 48)
    assert np.isfinite(fitted.in_sample_rmse)
    path = tmp_path / f"{item}.json"
    fitted.save(path)
    np.testing.assert_array_equal(
        FittedModel.load(path).predict_draws(x), fitted.predict_draws(x)
    )
    np.testing.assert_array_equal(
        pickle.loads(pickle.dumps(fitted)).predict_draws(x), fitted.predict_draws(x)
    )
    data = fitted.to_inference_data(x, y)
    assert set(data.children) >= {"posterior", "log_likelihood", "observed_data"}


@default_build
@pytest.mark.parametrize("item", items())
def test_every_row_reports_the_feature_without_it(item):
    x, _ = _fixture()
    model, y = ROWS[item]

    with pytest.raises(RequiresFeatureError):
        model.fit(x, y, random_state=1)


@pytest.mark.parametrize(
    "mean_params",
    [
        geometry(membership="hard"),
        structure("uniform"),
        TermParams(tessellations=8, cell=CellParams(basis="constant")),
    ],
)
def test_the_published_defaults_fit_in_every_build(mean_params, gaussian_fixture):
    x, y = gaussian_fixture

    fitted = Model(mean_params=mean_params, burn_in=10, draws=20).fit(
        x, y, random_state=1
    )

    assert fitted.n_draws == 20


def test_the_experimental_accessors_are_empty_where_nothing_is_sampled(
    gaussian_fixture,
):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1)

    assert fitted.dfs().size == 0
    assert fitted.cutpoints().size == 0
    assert fitted.bandwidths().size == 0
    assert fitted.inclusion_weights().size == 0
    assert fitted.concentrations().size == 0


@default_build
def test_predict_proba_reports_the_feature_without_it(gaussian_fixture):
    x, y = gaussian_fixture
    fitted = Model(**SMALL).fit(x, y, random_state=1)

    with pytest.raises(RequiresFeatureError):
        fitted.predict_proba(x)


@experimental
def test_a_survival_array_reaches_the_aft_family():
    x, _ = _fixture()
    model, y = ROWS["aft"]

    fitted = model.fit(x, y, random_state=1)

    assert fitted.model == "aft"
    assert "aft" in fitted.config["outcome"]
    assert fitted.sigma().shape == (20,)


@experimental
def test_the_aft_replicates_pair_with_the_observed_time():
    az = pytest.importorskip("arviz")
    x, _ = _fixture()
    model, y = ROWS["aft"]

    data = model.fit(x, y, random_state=1).to_inference_data(x, y)

    assert set(data["posterior_predictive"].dataset.data_vars) == {"time"}
    assert set(data["log_likelihood"].dataset.data_vars) == {"time"}
    assert {"time", "event"} <= set(data["observed_data"].dataset.data_vars)
    az.plot_ppc_dist(data, var_names=["time"], num_samples=10, backend="none")


@experimental
def test_the_interval_censored_replicates_keep_the_plain_name():
    pytest.importorskip("arviz")
    x, _ = _fixture()
    model, y = ROWS["interval_censored"]

    data = model.fit(x, y, random_state=1).to_inference_data(x, y)

    assert set(data["posterior_predictive"].dataset.data_vars) == {"y"}
    assert set(data["observed_data"].dataset.data_vars) == {"lower", "upper"}


@experimental
@pytest.mark.parametrize("item", ["aft", "interval_censored"])
def test_the_censored_families_pool_chains(item):
    x, _ = _fixture()
    model, y = ROWS[item]

    # Two short chains disagree, and the fit says so.
    with pytest.warns(UserWarning, match="may not have converged"):
        fitted = model.fit(x, y, random_state=1, n_chains=2, n_threads=2)

    assert fitted.n_draws == 40
    assert fitted.n_chains == 2


@experimental
def test_a_driven_aft_sampler_matches_fit_bit_for_bit():
    x, _ = _fixture()
    model, y = ROWS["aft"]
    through_fit = model.fit(x, y, random_state=1)

    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), random_state=1)
    sampler.step(10)
    for _ in range(20):
        sampler.step(1)
        sampler.keep()
    driven = sampler.finish()

    np.testing.assert_array_equal(driven.predict_draws(x), through_fit.predict_draws(x))
    assert driven.in_sample_rmse == through_fit.in_sample_rmse


@experimental
def test_set_response_takes_the_shape_of_the_family():
    x, y_numeric = _fixture()
    _, y = ROWS["aft"]
    sampler = Sampler(x, y, mean_params=TermParams(tessellations=8), random_state=1)

    with pytest.raises(ValueError, match="aft"):
        sampler.set_response(y_numeric)
    sampler.set_response(y)
    sampler.step(1)
    assert sampler.fitted_values().shape == (48,)


@experimental
def test_a_two_column_array_reaches_the_interval_censored_family():
    x, _ = _fixture()
    model, y = ROWS["interval_censored"]

    fitted = model.fit(x, y, random_state=1)

    assert fitted.model == "interval_censored"
    assert fitted.log_likelihood(x, y).shape == (20, 48)


@experimental
def test_an_ordered_categorical_reaches_the_ordinal_family():
    x, _ = _fixture()
    model, y = ROWS["ordinal"]

    fitted = model.fit(x, y, random_state=1)

    assert fitted.model == "ordinal"
    assert fitted.config["outcome"]["ordinal"]["categories"] == 3
    probs = fitted.predict_proba(x)
    assert probs.shape == (48, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)
    assert fitted.cutpoints().shape == (20, 1)
    assert fitted.sigma().size == 0


@experimental
def test_the_ordinal_category_count_is_checked_against_the_categories():
    x, _ = _fixture()
    _, y = ROWS["ordinal"]

    with pytest.raises(ValueError, match="2 categories but the response has 3"):
        Model(outcome=ordinal(categories=2), **SMALL).fit(x, y, random_state=1)


@experimental
def test_integer_codes_need_a_named_category_count():
    x, _ = _fixture()
    codes = np.arange(48) % 3

    with pytest.raises(ValueError, match="categories"):
        Model(outcome=ordinal(), **SMALL).fit(x, codes, random_state=1)
    fitted = Model(outcome=ordinal(categories=3), **SMALL).fit(x, codes, random_state=1)
    assert fitted.predict_proba(x).shape == (48, 3)


@experimental
@pytest.mark.parametrize(
    ("item", "variables"),
    [
        ("student_t_grid", {"df"}),
        ("ordinal", {"cutpoint"}),
        ("soft", {"bandwidth"}),
        ("dart", {"inclusion_weight", "concentration"}),
    ],
)
def test_the_posterior_carries_the_sampled_quantities(item, variables):
    x, _ = _fixture()
    model, y = ROWS[item]
    fitted = model.fit(x, y, random_state=1)

    posterior = fitted.to_inference_data(x, y)["posterior"].dataset

    assert variables <= set(posterior.data_vars)


@experimental
def test_the_accessors_have_one_row_per_draw():
    x, _ = _fixture()

    grid = ROWS["student_t_grid"][0].fit(x, ROWS["student_t_grid"][1], random_state=1)
    assert grid.dfs().shape == (20,)
    assert set(grid.dfs()) <= {3.0, 6.0, 12.0}

    soft = ROWS["soft"][0].fit(x, ROWS["soft"][1], random_state=1)
    assert soft.bandwidths().shape == (20, 8)

    dart = ROWS["dart"][0].fit(x, ROWS["dart"][1], random_state=1)
    assert dart.inclusion_weights().shape == (20, 2)
    np.testing.assert_allclose(dart.inclusion_weights().sum(axis=1), 1.0)
    assert dart.concentrations().shape == (20,)


@experimental
def test_the_observed_data_carries_the_censoring_columns():
    x, _ = _fixture()
    model, y = ROWS["aft"]
    fitted = model.fit(x, y, random_state=1)

    observed = fitted.to_inference_data(x, y)["observed_data"].dataset

    assert set(observed.data_vars) == {"time", "event"}
