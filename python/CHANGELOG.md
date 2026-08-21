# Changelog

Notable changes to the Python package, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Versions
follow semantic versioning. The sampled values for a fixed seed are those of
the core crate version the package builds against, which
`thiessen.CORE_VERSION` reports.

## [Unreleased]

### Added

- `Model`, the configuration, with every field of the core's `Config` as a
  keyword argument and `fit(X, y, random_state=None)`.
- `FittedModel`, the kept draws: `predict`, `predict_draws`,
  `predict_latent`, `predict_variance`, `predict_quantiles`,
  `credible_interval`, `prediction_interval`, `log_likelihood`, `sigma`,
  `cell_counts`, `dimension_counts`,
  `variable_inclusion_proportions`, and the resolved configuration, the
  fit-time warnings and the in-sample root mean squared error. Pickling
  through the core's serde representation.
- `ThiessenError`, a `ValueError`, carrying the core's message.
- `random_state` taking an `int`, a `numpy.random.Generator`, a
  `numpy.random.RandomState` or `None`, resolved by one rule; an integer
  passes through unchanged, and the resolved seed is on the fitted object.
- `CORE_VERSION`, the core crate version the extension was built from.
- `thiessen.estimators` with `AddiVortesRegressor` and
  `AddiVortesClassifier`, meeting the scikit-learn estimator contract:
  explicit `__init__` parameters, `n_features_in_`, `feature_names_in_`,
  `classes_`, `predict(X, return_std=False)`, `predict_proba`,
  `__sklearn_tags__`, `clone` and pickling. `scikit-learn` >= 1.6 is the
  `sklearn` extra.
- `categorical_features` on the estimators, of the
  `HistGradientBoostingRegressor` shape: `None`, `'from_dtype'`, an index
  array or a boolean mask. A named column becomes d - 1 treatment-contrast
  indicators, the first level as reference, unless its `metric` entry is
  `'categorical'`, in which case it passes as integer level codes.
