# thiessen 0.0.0.9000

* `thiessen()` fits the Gaussian, binary probit and heteroscedastic models
  to a numeric matrix, with `thiessen_control()` carrying the configuration
  in the shape the core stores it: an outcome family from
  `gaussian_outcome()` or `probit_outcome()`, one `term_params()` group per
  ensemble (with
  `geometry_params()` and `structure_params()` nested inside), and the
  sweep schedule from `general_params()`. A positive tessellation count on
  `variance_params` selects the heteroscedastic model, and
  `thiessen_control(tessellations = )` is the one shortcut, setting the
  mean ensemble's size. Methods: `predict()` for the
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
* `chains` runs that many chains, each with a seed the core derives from
  `seed`, and pools their draws. Two or more chains give rank-normalised
  split R-hat and the bulk and tail effective sample sizes of sigma and of
  the mean function at up to twenty training rows, through
  `posterior::summarise_draws()`. A fit warns, and `print()` and `summary()`
  repeat the warning, where R-hat exceeds 1.01 or an effective sample size
  falls below 400 (Vehtari and others, 2021); a fit of one chain says so
  instead.
* Five vignettes, executed at build: getting started, H-AddiVortes and
  Binary AddiVortes by their paper names, the control surface, and the
  sampler API with a worked censored-response imputation. A pkgdown site
  configuration groups the reference by surface.
* `thiessen_sampler()` (experimental) drives the core's Gibbs loop one
  call at a time: `$step(n)`, `$keep()`, `$set_response()`,
  `$fitted_values()`, `$noise_variances()`, `$finish()` returning the fit
  `thiessen()` returns. Burn-in and thinning are the caller's loop, the
  response may be replaced between sweeps, and driving the configured
  schedule by hand reproduces a one-chain fit bit for bit.
* `plot()` on a fit traces the per-draw diagnostics, one panel per
  quantity and one line per chain; distributional displays go through
  `posterior::as_draws_df()` and bayesplot.
* Conformance with the rOpenSci statistical software standards (general
  and Bayesian) is tagged through srr in `R/srr-stats-standards.R`, every
  standard met or waived with its reason.
* `thiessen_diagnostics()` returns the per-draw sigma, mean cells and mean
  active covariates, with the chain each draw comes from, and
  `variable_inclusion()` the share of the active dimensions falling on each
  covariate.
* Progress over the whole fit is signalled with progressr, so a session
  reports it after `progressr::handlers()` and nothing is printed by
  default. The sweeps, pooling the draws and the convergence summary each
  report, and each names itself in the progression's message, so the bar
  stands until the fit is complete. The draws do not depend on whether a
  handler is set.
* A fit is a plain R object holding the sampler state, so `saveRDS()`
  writes one and a later session reads it without a refit.
* The call is stored on a fit, so `stats::update()` refits with an argument
  replaced.
* `seed = NULL` draws the chain's seed from R's stream, so `set.seed()`
  governs; a whole number passes to the core unchanged.
* The core is built without its `experimental` feature, so only the
  published models are reachable: a gated outcome has no constructor and a
  configuration naming a gated field or variant is rejected with the
  core's message; `core_experimental()` reports the build's setting.
* Errors carry the condition class `thiessen_error` and warnings
  `thiessen_warning`.
* Builds and links the core crate offline from vendored sources, the core
  as source under `src/rust/core` and the third-party crates in
  `src/rust/vendor.tar.xz`; `core_version()` reports the core version.
