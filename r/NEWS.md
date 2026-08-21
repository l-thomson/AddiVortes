# thiessen 0.0.0.9000

* `thiessen()` fits the Gaussian, binary probit and heteroscedastic models
  to a numeric matrix, with `thiessen_control()` carrying the
  hyperparameters and the sweep schedule. Methods: `predict()` for the
  posterior mean, the per-draw mean function and variance, and central
  credible and posterior predictive intervals; `sigma()`, `fitted()`,
  `residuals()`, `nobs()`, `print()` and `summary()`.
* `thiessen(formula, data)` and `thiessen(data.frame, y)` through hardhat, so
  `predict(newdata = )` matches columns by name and type and reports a
  missing one. A factor covariate becomes d - 1 treatment-contrast
  indicators, the first level as reference; where `control` declares a
  `metric`, factors pass as integer level codes and each factor column must
  declare `"categorical"`. A two-level factor response becomes 0 and 1 with
  the first level as the zero.
* The call is stored on a fit, so `stats::update()` refits with an argument
  replaced.
* `seed = NULL` draws the chain's seed from R's stream, so `set.seed()`
  governs; a whole number passes to the core unchanged.
* Errors carry the condition class `thiessen_error` and warnings
  `thiessen_warning`.
* Builds and links the core crate offline from the vendored sources in
  `src/rust/vendor.tar.xz`; `core_version()` reports the core version.
