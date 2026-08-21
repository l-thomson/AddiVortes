"""The scikit-learn estimators."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from thiessen import Model, ThiessenError
from thiessen._config import FIELDS
from thiessen.estimators import (
    _NON_CONFIG,
    AddiVortesClassifier,
    AddiVortesRegressor,
)

from .conftest import SMALL


def test_parameters_are_configuration_fields():
    for estimator in (AddiVortesRegressor(), AddiVortesClassifier()):
        names = set(estimator.get_params())
        assert names - _NON_CONFIG <= set(FIELDS)


def test_regressor_fits_and_predicts(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL).fit(x, y)

    check_is_fitted(model)
    assert model.n_features_in_ == 2
    assert model.random_state_ == 1
    assert model.predict(x).shape == (48,)
    assert model.score(x, y) > 0.0


def test_regressor_return_std(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL).fit(x, y)

    mean, std = model.predict(x, return_std=True)
    np.testing.assert_allclose(mean, model.predict(x))
    assert std.shape == (48,)
    assert np.all(std > 0.0)


def test_regressor_prediction_interval_brackets_the_mean(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL).fit(x, y)

    interval = model.predict_interval(x, level=0.9)
    mean = model.predict(x)
    assert np.all(interval[:, 0] <= mean)
    assert np.all(mean <= interval[:, 1])


def test_heteroscedastic_model(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(
        model="heteroscedastic", m_var=5, random_state=1, **SMALL
    ).fit(x, y)

    assert model.predict(x).shape == (48,)


def test_probit_is_rejected_by_the_regressor(gaussian_fixture):
    x, y = gaussian_fixture
    with pytest.raises(ValueError, match="AddiVortesClassifier"):
        AddiVortesRegressor(model="probit", **SMALL).fit(x, y)


def test_unknown_model_is_rejected(gaussian_fixture):
    x, y = gaussian_fixture
    with pytest.raises(ValueError, match="unknown model"):
        AddiVortesRegressor(model="soft", **SMALL).fit(x, y)


def test_classifier_predicts_labels_and_probabilities(probit_fixture):
    x, y = probit_fixture
    model = AddiVortesClassifier(random_state=1, **SMALL).fit(x, y)

    np.testing.assert_array_equal(model.classes_, [0.0, 1.0])
    probabilities = model.predict_proba(x)
    assert probabilities.shape == (48, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert set(np.unique(model.predict(x))) <= {0.0, 1.0}
    assert model.score(x, y) > 0.5


def test_classifier_keeps_the_caller_labels(probit_fixture):
    x, y = probit_fixture
    labels = np.where(y > 0.5, "high", "low")

    model = AddiVortesClassifier(random_state=1, **SMALL).fit(x, labels)

    np.testing.assert_array_equal(model.classes_, ["high", "low"])
    assert set(np.unique(model.predict(x))) <= {"high", "low"}


def test_classifier_rejects_more_than_two_classes(gaussian_fixture):
    x, _ = gaussian_fixture
    three = np.array([i % 3 for i in range(48)])
    with pytest.raises(ValueError, match="Only binary classification"):
        AddiVortesClassifier(**SMALL).fit(x, three)


def test_the_estimator_and_model_agree_for_one_seed(gaussian_fixture):
    x, y = gaussian_fixture

    estimator = AddiVortesRegressor(random_state=1, **SMALL).fit(x, y)
    direct = Model(**SMALL).fit(x, y, random_state=1)

    np.testing.assert_array_equal(estimator.predict(x), direct.predict(x))


def test_clone_and_get_params_round_trip():
    model = AddiVortesRegressor(lambda_c=25.0, m=10)
    copy = clone(model)

    assert copy.get_params() == model.get_params()
    assert copy.lambda_c == 25.0


def test_pickle_round_trip_preserves_predictions(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL).fit(x, y)

    restored = pickle.loads(pickle.dumps(model))

    np.testing.assert_array_equal(restored.predict(x), model.predict(x))
    assert restored.random_state_ == model.random_state_


def test_pickle_of_an_unfitted_estimator(gaussian_fixture):
    model = AddiVortesRegressor(m=10)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.get_params() == model.get_params()


def test_predict_before_fit_is_rejected(gaussian_fixture):
    x, _ = gaussian_fixture
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        AddiVortesRegressor().predict(x)


def test_predict_with_the_wrong_column_count(gaussian_fixture):
    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL).fit(x, y)
    with pytest.raises(ValueError, match="features"):
        model.predict(x[:, :1])


def test_core_errors_surface_as_value_errors(gaussian_fixture):
    x, y = gaussian_fixture
    x = x.copy()
    x[:, 1] = 1.0
    with pytest.raises(ThiessenError):
        AddiVortesRegressor(**SMALL).fit(x, y)


def test_feature_names_are_stored():
    pandas = pytest.importorskip("pandas")
    frame = pandas.DataFrame(
        {"a": np.linspace(0.0, 1.0, 30), "b": np.linspace(1.0, 0.0, 30)}
    )
    y = frame["a"].to_numpy() + frame["b"].to_numpy() * 0.5

    model = AddiVortesRegressor(random_state=1, **SMALL).fit(frame, y)

    np.testing.assert_array_equal(model.feature_names_in_, ["a", "b"])


def test_pipeline_and_cross_validation(gaussian_fixture):
    x, y = gaussian_fixture
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("fit", AddiVortesRegressor(random_state=1, **SMALL)),
        ]
    )

    scores = cross_val_score(pipeline, x, y, cv=3)

    assert scores.shape == (3,)


def test_grid_search(gaussian_fixture):
    x, y = gaussian_fixture
    search = GridSearchCV(
        AddiVortesRegressor(random_state=1, **SMALL),
        {"lambda_c": [5.0, 25.0]},
        cv=2,
    )

    search.fit(x, y)

    assert search.best_params_["lambda_c"] in (5.0, 25.0)


def test_partial_dependence_works_through_sklearn(gaussian_fixture):
    from sklearn.inspection import partial_dependence

    x, y = gaussian_fixture
    model = AddiVortesRegressor(random_state=1, **SMALL).fit(x, y)

    result = partial_dependence(model, x, features=[0], grid_resolution=5)

    assert result["average"].shape == (1, 5)


class TestCategorical:
    """The `categorical_features` parameter."""

    @staticmethod
    def _data():
        n = 60
        codes = np.array([i % 3 for i in range(n)], dtype=float)
        continuous = np.linspace(0.0, 1.0, n)
        x = np.column_stack([continuous, codes])
        y = continuous + 0.5 * codes
        return x, y

    def test_indices_expand_to_treatment_contrasts(self):
        x, y = self._data()
        model = AddiVortesRegressor(
            categorical_features=[1], random_state=1, **SMALL
        ).fit(x, y)

        # Three levels become two indicator columns, the first as reference.
        assert model._encoding.core_metric == ["euclidean"] * 3
        assert model.n_features_in_ == 2
        assert model.predict(x).shape == (60,)

    def test_boolean_mask_is_accepted(self):
        x, y = self._data()
        model = AddiVortesRegressor(
            categorical_features=[False, True], random_state=1, **SMALL
        ).fit(x, y)

        assert model.predict(x).shape == (60,)

    def test_eskin_metric_passes_integer_codes(self):
        x, y = self._data()
        model = AddiVortesRegressor(
            categorical_features=[1],
            metric=["euclidean", "categorical"],
            random_state=1,
            **SMALL,
        ).fit(x, y)

        assert model._encoding.core_metric == ["euclidean", "categorical"]
        assert model.predict(x).shape == (60,)

    def test_no_encoding_happens_without_the_parameter(self):
        x, y = self._data()
        model = AddiVortesRegressor(random_state=1, **SMALL).fit(x, y)

        assert model._encoding.core_metric == ["euclidean", "euclidean"]

    def test_string_levels_are_encoded(self):
        n = 60
        continuous = np.linspace(0.0, 1.0, n)
        letters = np.array(["a", "b", "c"] * (n // 3))
        x = np.empty((n, 2), dtype=object)
        x[:, 0] = continuous
        x[:, 1] = letters
        y = continuous + 0.5 * (letters == "b")

        model = AddiVortesRegressor(
            categorical_features=[1], random_state=1, **SMALL
        ).fit(x, y)

        assert model.predict(x).shape == (60,)

    def test_an_unseen_level_is_rejected(self):
        x, y = self._data()
        model = AddiVortesRegressor(
            categorical_features=[1], random_state=1, **SMALL
        ).fit(x, y)

        unseen = x.copy()
        unseen[0, 1] = 9.0
        with pytest.raises(ValueError, match="not one of the levels"):
            model.predict(unseen)

    def test_a_single_level_is_rejected(self):
        x, y = self._data()
        constant = x.copy()
        constant[:, 1] = 0.0
        with pytest.raises(ValueError, match="at least two"):
            AddiVortesRegressor(categorical_features=[1], **SMALL).fit(constant, y)

    def test_an_out_of_range_index_is_rejected(self):
        x, y = self._data()
        with pytest.raises(ValueError, match="outside"):
            AddiVortesRegressor(categorical_features=[5], **SMALL).fit(x, y)

    def test_an_unknown_string_is_rejected(self):
        x, y = self._data()
        with pytest.raises(ValueError, match="from_dtype"):
            AddiVortesRegressor(categorical_features="auto", **SMALL).fit(x, y)

    def test_from_dtype_reads_the_pandas_dtypes(self):
        pandas = pytest.importorskip("pandas")
        n = 60
        frame = pandas.DataFrame(
            {
                "x": np.linspace(0.0, 1.0, n),
                "g": pandas.Categorical(["a", "b", "c"] * (n // 3)),
            }
        )
        y = frame["x"].to_numpy() + 0.5 * (frame["g"] == "b")

        model = AddiVortesRegressor(
            categorical_features="from_dtype", random_state=1, **SMALL
        ).fit(frame, y)

        assert model._encoding.core_metric == ["euclidean"] * 3
        np.testing.assert_array_equal(model.feature_names_in_, ["x", "g"])
        assert model.predict(frame).shape == (60,)

    def test_from_dtype_needs_a_data_frame(self):
        x, y = self._data()
        with pytest.raises(ValueError, match="data frame"):
            AddiVortesRegressor(categorical_features="from_dtype", **SMALL).fit(x, y)
