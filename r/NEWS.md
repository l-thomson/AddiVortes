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
* Methods on the established generics: `posterior::as_draws_df()`,
  `as_draws_array()`, `ndraws()` and `nchains()`, registered on demand so
  posterior is needed only by a session that calls them, with R-hat and
  effective sample sizes coming from `posterior::summarise_draws()`; and
  `posterior_predict()`, `posterior_epred()`, `log_lik()` and
  `predictive_interval()` from rstantools, re-exported so no
  `library(rstantools)` is needed. `loo::loo()` runs on `log_lik()`. The
  draws carry `mu[i]`, `sigma` under the Gaussian model only, and the
  per-draw `cell_count` and `dimension_count`.
* The call is stored on a fit, so `stats::update()` refits with an argument
  replaced.
* `seed = NULL` draws the chain's seed from R's stream, so `set.seed()`
  governs; a whole number passes to the core unchanged.
* Errors carry the condition class `thiessen_error` and warnings
  `thiessen_warning`.
* Builds and links the core crate offline from the vendored sources in
  `src/rust/vendor.tar.xz`; `core_version()` reports the core version.
