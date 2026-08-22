# Changelog

Notable changes to the Python package, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format. Versions
follow semantic versioning. The sampled values for a fixed seed are those of
the core crate version the package builds against, which
`thiessen.CORE_VERSION` reports.

## [Unreleased]

### Added

- `Model`, the configuration, in the shape the core stores it: an outcome
  family from `gaussian(nu, q)` or `probit(offset)`, one `TermParams` group
  per ensemble (`mean_params`, `variance_params`), the flat run-length
  settings, and `fit(X, y, random_state=None, n_chains=1)`. A positive
  tessellation count on `variance_params` selects the heteroscedastic
  model. The family and group objects implement `get_params`/`set_params`
  with `<group>__<parameter>` routing, compare by value and reproduce
  their constructor call in `repr`.
- `n_chains` runs that many chains, each with a seed the core derives from
  the resolved seed, and pools their draws; the chain dimension of
  `to_inference_data` holds them. Two or more chains warn where R-hat
  exceeds 1.01 or a bulk or tail effective sample size falls below 400
  (Vehtari and others, 2021), computed by arviz on sigma and on the mean
  function at up to twenty training rows; a fit says so where arviz is not
  installed. `AddiVortesRegressor` and `AddiVortesClassifier` take
  `n_chains` as a constructor parameter.
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
  `__sklearn_tags__`, `clone` and pickling. The configuration groups are
  the same objects as on `Model`, so grid search routes into them:
  `GridSearchCV(model, {"mean_params__tessellations": [50, 200]})`.
  `scikit-learn` >= 1.6 is the `sklearn` extra.
- `categorical_features` on the estimators, of the
  `HistGradientBoostingRegressor` shape: `None`, `'from_dtype'`, an index
  array or a boolean mask. A named column becomes d - 1 treatment-contrast
  indicators, the first level as reference, unless its entry in the
  geometry's `metric` is `'categorical'`, in which case it passes as
  integer level codes.
- Coverage reporting for the Python suite, uploaded under the `python` flag.
- Packaging metadata completed against the pyOpenSci guide: the issue
  tracker, the platform and language classifiers, and a statement of need,
  contributing and citation sections in the README.
- A documentation site built with mkdocs-material and mkdocstrings: quick
  start, the model menu with parameter tables, priors and scaling with the
  BART correspondence table, the input-data contract, the scikit-learn and
  arviz pages, reproducibility, and the shared testing strategy. Every code
  example is executed at build and `mkdocs build --strict` runs in CI. The
  site is a CI artefact in this release; it is not deployed.
- `FittedModel.to_inference_data(X, y)`, returning the arviz `DataTree` of
  the fit: `posterior` with `mu` per draw, `sigma` under the Gaussian model
  only, and the per-draw cell and dimension counts; `posterior_predictive`
  and `log_likelihood` with `y`; and `observed_data`. The chain dimension is
  one, as the sampler runs one chain, and the predictive replicates are drawn
  in numpy from the fit's resolved seed rather than by the core. `arviz>=1.0`
  is the `arviz` extra; the groups are built through `arviz.from_dict` with
  no version dispatch.
- `FittedModel.save` and `FittedModel.load`, taking `str` or `os.PathLike`
  and writing the core's serde representation, which reloads bit-exact.
  Failures to read or write are `OSError`; contents that are not a fitted
  model are `ThiessenError`.
- The core's fit-time warnings are raised as `UserWarning` at fit, by `Model`
  and by the estimators, as well as staying on `FittedModel.warnings`.
- Overloads on `AddiVortesRegressor.predict`, so `return_std=True` types as a
  pair of arrays rather than a union.
- `numpy.integer` accepted for `random_state`, which already worked at
  runtime.
- `_native.EXPERIMENTAL`, whether the extension was built with the core's
  `experimental` feature. It is off in every wheel of this release, so the
  published models are the only ones reachable: a gated outcome has no
  constructor in the package and a configuration naming a gated field or
  variant is rejected by the core, so an item added to or graduated from
  the feature needs no change here.
