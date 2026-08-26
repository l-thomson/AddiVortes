# Conformance with the rOpenSci statistical software standards (general and
# Bayesian). Each tag states where the standard is met; the NA block at the
# end states why the remaining standards do not apply. srr::srr_report()
# checks that every standard of both categories appears exactly once.

#' srr_stats
#'
#' @srrstatsVerbose FALSE
#'
#' @srrstats {G1.0} DESCRIPTION, `thiessen()` and README cite Stone and
#'   Gosling (2025, \doi{10.1080/10618600.2024.2414104}) as the primary
#'   reference, and Chipman, George and McCulloch (2010) for the BART
#'   framework the method extends.
#' @srrstats {G1.1} README states that the package is an independent
#'   implementation of the published algorithm and links the authors' R
#'   package, against which the test suite compares posterior summaries
#'   (docs/testing.md, section Upstream comparison).
#' @srrstats {G1.2} CONTRIBUTING.md, section Stable and experimental, is
#'   the life cycle statement: the stable surface is the published models,
#'   and everything else carries the experimental tier.
#'   `thiessen_sampler()` and the experimental outcome families carry a
#'   lifecycle badge.
#' @srrstats {G1.3} Statistical terms (tessellation, cell, burn-in,
#'   thinning, kept draw, R-hat, effective sample size) are defined where
#'   first used in `thiessen()`, `thiessen_control()` and the vignettes.
#' @srrstats {G1.4} Every exported function is documented with roxygen2.
#' @srrstats {G1.4a} Every internal function is documented with roxygen2
#'   and `@noRd`.
#' @srrstats {G2.0} Lengths are asserted at the boundary: `check_number()`
#'   and its siblings for every scalar option, the design and response row
#'   check in `new_fit()`, and the per-column length check on `metric` in
#'   the core.
#' @srrstats {G2.0a} The documented types in `thiessen()`,
#'   `thiessen_control()` and the parameter groups state the expected
#'   lengths, for example `y` as a vector of `nrow(x)` values.
#' @srrstats {G2.1} Types are asserted at the boundary: `as_design()`,
#'   `as_response()`, `check_number()` and `check_group()`, with the core's
#'   deserialiser as a second gate on the whole configuration.
#' @srrstats {G2.1a} Every `@param` entry states the expected type.
#' @srrstats {G2.2} Scalar options reject vectors through `check_number()`,
#'   `check_whole_number()`, `check_flag()` and `check_probability()`,
#'   which wrap rlang's input checks under the package's condition class;
#'   tested in test-control.R and test-determinism.R.
#' @srrstats {G2.3} Character input is restricted to the `metric` entries.
#' @srrstats {G2.3a} The core's deserialiser permits only the documented
#'   metric values and errors with the field and the offending value;
#'   tested in test-control.R.
#' @srrstats {G2.3b} `geometry_params()` documents that metric entries are
#'   matched case-sensitively.
#' @srrstats {G2.4} Conversions at the boundary are explicit, as itemised
#'   below.
#' @srrstats {G2.4a} `chains` is converted with `as.integer()` after a
#'   whole-number check.
#' @srrstats {G2.4b} Designs and responses are converted to double with
#'   `storage.mode()` and `as.double()`.
#' @srrstats {G2.4e} A factor covariate or response is decoded with
#'   `as.integer()` on its levels, as documented in `thiessen()`.
#' @srrstats {G2.5} `thiessen()` documents that a factor is encoded by its
#'   level codes or treatment contrasts and that an ordered factor is
#'   treated as unordered.
#' @srrstats {G2.6} A one-dimensional input is taken as a one-column
#'   design by `as_design()` regardless of extra attributes; a numeric
#'   column carrying foreign attributes fits as its values (tested in
#'   test-fit.R).
#' @srrstats {G2.7} Input is accepted as a matrix, as a data frame
#'   (including tibbles, through `hardhat::mold()`) and through a formula;
#'   test-frame.R shows the three agree.
#' @srrstats {G2.8} All input paths converge on one double matrix and one
#'   double vector before the core is called (`encode_predictors()`,
#'   `encode_response()`, `as_design()`, `as_response()`).
#' @srrstats {G2.9} No conversion loses information silently: the factor
#'   encoding is documented in `thiessen()` and asserted in test-frame.R,
#'   column names are kept, and no names are invented for unnamed input
#'   (`variable_inclusion()` returns unnamed values in that case).
#' @srrstats {G2.10} Columns are extracted with `[[` on molded frames and
#'   with `drop = FALSE` on matrices throughout.
#' @srrstats {G2.11} A data frame column with foreign attributes or extra
#'   classes fits as its underlying values; tested in test-fit.R.
#' @srrstats {G2.12} A list column is refused with an error naming the
#'   variable; tested in test-fit.R.
#' @srrstats {G2.13} Missing values are checked before any computation:
#'   `as_design()` and `as_response()` reject `NA`, and the core rejects
#'   non-finite values with their position; tested in test-fit.R.
#' @srrstats {G2.14} The package takes one documented position on missing
#'   data, as itemised below.
#' @srrstats {G2.14a} Missing data errors, with the condition class
#'   `thiessen_error`; `thiessen()` documents that no row is dropped
#'   silently.
#' @srrstats {G2.15} No internal computation touches user data before the
#'   missingness and finiteness checks above, so `na.rm`-style defaults
#'   are never relied on.
#' @srrstats {G2.16} Non-finite values (`NaN`, `Inf`, `-Inf`) are rejected
#'   by the core with an error naming the row and column; tested in
#'   test-fit.R. Silently ignoring them is not offered: the posterior has
#'   no meaning for such input.
#' @srrstats {G3.0} No floating-point value is compared for equality
#'   outside the determinism suite, where bit identity against a stored
#'   chain is itself the property under test (test-determinism.R); all
#'   statistical tests compare within stated tolerances (test-recovery.R).
#' @srrstats {G5.1} Every test data set is constructed by short documented
#'   code in the test files and helper-fixture.R, which ship in the
#'   package, so any user can reproduce them.
#' @srrstats {G5.2} Error and warning behaviour is tested throughout
#'   test-control.R, test-fit.R, test-frame.R and test-convergence.R.
#' @srrstats {G5.2a} Condition messages are composed per call site and
#'   are distinct; each names its argument or field.
#' @srrstats {G5.2b} The tests above trigger the conditions and match
#'   their messages and classes.
#' @srrstats {G5.3} test-recovery.R asserts that fitted values, posterior
#'   expectations and sigma draws are finite.
#' @srrstats {G5.4} Correctness is tested at three levels: bit-exact
#'   reproduction of chains the core commits (test-determinism.R),
#'   known-answer and simulation-recovery tests in the vendored core, and
#'   comparison of posterior summaries against the authors' R package
#'   (docs/testing.md).
#' @srrstats {G5.4a} The method is published; correctness of this
#'   implementation is separated from correctness of the method by the
#'   upstream comparison and the calibration suite (docs/testing.md).
#' @srrstats {G5.4b} The upstream comparison runs against the CRAN
#'   AddiVortes package; the stored chains in tests/testthat/core-*.txt
#'   are fixed outputs the R package must reproduce bit for bit.
#' @srrstats {G5.4c} The stored chains are committed text files, checked
#'   against the vendored core's copies in CI.
#' @srrstats {G5.5} Every correctness test fixes its seed.
#' @srrstats {G5.6} test-recovery.R recovers a known signal with in-sample
#'   error below a stated tolerance.
#' @srrstats {G5.6a} The recovery tests state their tolerances next to the
#'   comparison.
#' @srrstats {G5.6b} The recovery test runs under three seeds and bounds
#'   the spread of the posterior means.
#' @srrstats {G5.7} The calibration suite runs at a small and a full size
#'   (docs/testing.md, section Calibration), so miscalibration that only
#'   shows at scale is caught by the nightly job.
#' @srrstats {G5.8} Edge conditions error with informative messages, as
#'   itemised below; all are tested in test-fit.R.
#' @srrstats {G5.8a} Zero-length data is rejected by the core with the
#'   observation count in the message.
#' @srrstats {G5.8b} Unsupported types (data frames where a matrix is
#'   required, non-numeric vectors, factors of more than two levels) are
#'   rejected; tested in test-fit.R and test-frame.R.
#' @srrstats {G5.8c} Data with `NA` fields is rejected, and a constant
#'   column is rejected with the column named.
#' @srrstats {G5.8d} A design with more covariates than observations
#'   warns; tested in test-fit.R.
#' @srrstats {G5.9} Noise susceptibility is tested in test-recovery.R, as
#'   itemised below.
#' @srrstats {G5.9a} Perturbing the design at machine precision moves the
#'   posterior mean less than the stated tolerance.
#' @srrstats {G5.9b} Three seeds give posterior means within the stated
#'   tolerance of each other, while test-determinism.R shows distinct
#'   seeds give distinct draws.
#' @srrstats {G5.10} The extended tier is the full-size calibration suite,
#'   switched on with `cargo nextest run --run-ignored all` and run by the
#'   nightly calibration job (docs/testing.md, section Running the full
#'   suite locally).
#' @srrstats {G5.12} docs/testing.md states the sizes, levels and runtime
#'   expectations of the extended suite and how to run it.
#' @srrstats {BS1.1} `thiessen()` documents entry of the design and the
#'   response for the matrix, data frame and formula interfaces, each with
#'   an executable example; the vignette thiessen.Rmd walks a fit.
#' @srrstats {BS1.2} Prior specification is described at the three levels
#'   itemised below.
#' @srrstats {BS1.2a} README, section Priors, describes every prior and
#'   its parameter.
#' @srrstats {BS1.2b} The vignette control-surface.Rmd describes the
#'   priors in general and applied terms with executable code.
#' @srrstats {BS1.2c} `thiessen_control()`, `gaussian_outcome()`,
#'   `term_params()`, `geometry_params()` and `structure_params()` document
#'   each parameter, with examples.
#' @srrstats {BS1.3} `general_params()` documents burn-in, draw count and
#'   thinning; `thiessen()` documents `chains` and `seed`.
#' @srrstats {BS1.3a} The vignette sampler-api.Rmd shows a run continued
#'   one sweep at a time and finished into an ordinary fit, which is how a
#'   previous simulation seeds further sampling.
#' @srrstats {BS1.4} `thiessen()` documents and its examples show a fit
#'   with the convergence diagnostics (`chains = 2`) and without (one
#'   chain); the vignette thiessen.Rmd repeats this.
#' @srrstats {BS2.1} The design and response row counts must agree
#'   (`new_fit()`), and `metric` must have one entry per column (core
#'   check).
#' @srrstats {BS2.1a} Tested in test-fit.R ("the design and the response
#'   must agree") and test-control.R.
#' @srrstats {BS2.2} The whole configuration is validated at
#'   `thiessen_control()` time, before any chain runs; tested in
#'   test-control.R.
#' @srrstats {BS2.3} Distributional parameters are scalars enforced by
#'   `check_number()`; nothing is discarded silently.
#' @srrstats {BS2.4} The one vector option, `metric`, must match the
#'   design's column count; the core errors otherwise.
#' @srrstats {BS2.5} The core validates positivity and ranges (`nu`, `k`,
#'   `lambda_c`, `sigma_c`, `omega` positive; `q` a probability); tested in
#'   test-control.R ("the core rejects an invalid value at construction").
#' @srrstats {BS2.6} `tessellations`, `burn_in`, `draws`, `thinning`,
#'   `chains` and `seed` are checked as whole numbers in their documented
#'   ranges by `check_whole_number()`; tested in test-control.R and
#'   test-determinism.R.
#' @srrstats {BS2.8} `thiessen_sampler()` keeps a run alive: the caller
#'   steps further, keeps more draws and finishes later, so a previous
#'   run's state is the starting point of the next stretch.
#' @srrstats {BS2.9} Each chain's seed is derived in the core from `seed`
#'   and the chain index, so chains never share a stream; tested in
#'   test-chains.R.
#' @srrstats {BS2.12} Errors and warnings always signal; progress follows
#'   the progressr convention, where the session opts in with
#'   `progressr::handlers()`, so nothing is printed by default.
#' @srrstats {BS2.13} Progress is silent unless a handler is set, while
#'   warnings still signal; tested in test-progress.R.
#' @srrstats {BS2.14} Warnings carry the condition class
#'   `thiessen_warning`, so they can be suppressed selectively; tested in
#'   test-convergence.R and test-fit.R.
#' @srrstats {BS2.15} Errors carry the condition class `thiessen_error`
#'   and are trappable with `tryCatch()`; tested across the suite. An
#'   error naming the core's `experimental` feature carries
#'   `thiessen_requires_feature` before it; tested in
#'   test-experimental.R.
#' @srrstats {BS3.0} `thiessen()` documents that missing and non-finite
#'   values are rejected and that no row is dropped silently.
#' @srrstats {BS4.0} The Gibbs backfitting sampler with
#'   Metropolis-Hastings moves is documented in `thiessen()` with the
#'   citation of Stone and Gosling (2025), and in docs/models.md next to
#'   the vendored core.
#' @srrstats {BS4.1} The test suite compares posterior summaries against
#'   the authors' CRAN package (docs/testing.md, section Upstream
#'   comparison).
#' @srrstats {BS4.2} Posterior validity is tested by simulation-based
#'   calibration and Geweke tests in the vendored core, with the gates
#'   shown to reject a mispriced sampler (docs/testing.md, sections
#'   Calibration and Tests of tests).
#' @srrstats {BS4.3} Rank-normalised split R-hat and bulk and tail
#'   effective sample sizes are computed with posterior for two or more
#'   chains, with Vehtari and others (2021) cited in `thiessen()`.
#' @srrstats {BS4.4} `thiessen_sampler()` lets the caller step, inspect
#'   and stop on any criterion, including a convergence check on the kept
#'   draws so far.
#' @srrstats {BS4.5} A fit whose diagnostics cross the documented
#'   thresholds warns, and `print()` and `summary()` repeat the warning;
#'   tested in test-convergence.R.
#' @srrstats {BS4.6} The diagnostics are computed after sampling and do
#'   not alter the draws: test-chains.R shows the first chain of a
#'   two-chain fit equals the single-chain fit bit for bit.
#' @srrstats {BS5.0} The fit stores `seed` and `n_chains`; per-chain seeds
#'   are derived from them as `thiessen()` documents. Tested in
#'   test-determinism.R ("the seed used is stored").
#' @srrstats {BS5.1} The fit stores the design, the response, the column
#'   count and the hardhat blueprint where one applies; `nobs()` reports
#'   the observation count.
#' @srrstats {BS5.2} The fit stores the resolved configuration, priors
#'   included, printed by `print()` and tested in test-fit.R ("the
#'   resolved configuration is on the fit").
#' @srrstats {BS5.3} The fit stores the convergence diagnostics for two or
#'   more chains (`convergence_of()`); tested in test-convergence.R.
#' @srrstats {BS5.5} `print()` and `summary()` show the diagnostics and
#'   repeat the non-convergence warning; tested in test-convergence.R.
#' @srrstats {BS6.0} `print.thiessen()`.
#' @srrstats {BS6.1} `plot.thiessen()`.
#' @srrstats {BS6.2} `plot.thiessen()` plots the kept-draw traces per
#'   chain and its documentation states that burn-in sweeps are discarded
#'   before the first draw is kept; `thiessen_diagnostics()` and
#'   `posterior::as_draws_df()` feed `bayesplot::mcmc_trace()` for the
#'   mean function.
#' @srrstats {BS6.3} The `plot.thiessen()` documentation points
#'   distributional displays at `bayesplot::mcmc_areas()` and
#'   `mcmc_dens()` on `posterior::as_draws_df()`; test-generics.R shows
#'   bayesplot takes the draws unmodified.
#' @srrstats {BS6.4} `summary.thiessen()`.
#' @srrstats {BS6.5} `bayesplot::mcmc_combo()` on
#'   `posterior::as_draws_df()` plots sequences and densities together, as
#'   the `plot.thiessen()` documentation states.
#' @srrstats {BS7.0} `prior_only = TRUE` in `general_params()` samples the
#'   prior, and the simulation-based calibration suite draws parameters
#'   from the prior and checks rank uniformity (docs/testing.md).
#' @srrstats {BS7.1} The prior-only chain is the no-data case; the SBC
#'   gates confirm the prior is reproduced in distribution.
#' @srrstats {BS7.2} test-recovery.R recovers a known signal from data,
#'   and the Geweke and upstream-comparison suites check the posterior
#'   against independent derivations (docs/testing.md).
#' @srrstats {BS7.4} Fitted and predicted values are on the response
#'   scale: test-recovery.R checks recovery on a response three orders of
#'   magnitude from the covariates, and test-methods.R checks
#'   `predict()` equals the fitted values at the training rows.
#' @srrstats {BS7.4a} The core standardises the response internally and
#'   maps back through a frozen affine map; test-recovery.R shows a
#'   response far from zero keeps its scale and mean.
#' @noRd
NULL

#' NA_standards
#'
#' @srrstatsNA {G1.5} The package makes no performance claims in an
#'   associated publication; correctness is compared against the reference
#'   implementation instead.
#' @srrstatsNA {G1.6} As G1.5: no performance claims are made, so there is
#'   no claim-comparison code to include.
#' @srrstatsNA {G2.4c} No input is converted to character; the only
#'   character values are the fixed metric names.
#' @srrstatsNA {G2.4d} No input is converted to factor; factors are
#'   accepted and decoded, never created.
#' @srrstatsNA {G2.14b} Silently ignoring rows with missing data would
#'   change the posterior without notice, so it is not offered; missing
#'   data errors instead.
#' @srrstatsNA {G2.14c} Imputation is model-specific, so no generic option
#'   is offered; the vignette sampler-api.Rmd shows imputation of censored
#'   responses through `thiessen_sampler()`.
#' @srrstatsNA {G3.1} The package computes no covariances; sigma^2 is
#'   sampled, not estimated from `stats::cov()`.
#' @srrstatsNA {G3.1a} As G3.1: no covariance method exists to document.
#' @srrstatsNA {G4.0} No function writes local files; persistence is the
#'   user's `saveRDS()` on the fit, as documented in `thiessen()`.
#' @srrstatsNA {G5.0} No standard reference data set exercises a Bayesian
#'   tessellation sampler; tests simulate from stated generative models
#'   and compare against the reference implementation instead.
#' @srrstatsNA {G5.11} The extended tests need no downloaded assets; every
#'   fixture is generated in the test code.
#' @srrstatsNA {G5.11a} As G5.11: there are no downloads to fail.
#' @srrstatsNA {BS1.0} The documentation does not use the term
#'   "hyperparameter"; every prior parameter is named individually.
#' @srrstatsNA {BS1.3b} The package implements the one sampler of the
#'   paper; there is no alternative algorithm to select.
#' @srrstatsNA {BS1.5} Only one convergence checker is implemented, so
#'   there are no differences between checkers to test.
#' @srrstatsNA {BS2.7} Every chain starts at the paper's deterministic
#'   initial ensemble; the seed is the only stochastic control, so there
#'   are no user-set starting values.
#' @srrstatsNA {BS2.10} The interface takes one seed and derives distinct
#'   per-chain seeds in the core, so identical seeds cannot reach distinct
#'   chains.
#' @srrstatsNA {BS2.11} There is no starting-value parameter to name.
#' @srrstatsNA {BS3.1} The sampler never inverts the design, so perfect
#'   collinearity is valid input rather than a degeneracy: as in BART,
#'   collinear covariates share inclusion under the covariate prior.
#' @srrstatsNA {BS3.2} As BS3.1: no distinct routine is needed because the
#'   sampling algorithm is unchanged by collinearity.
#' @srrstatsNA {BS4.7} The checker takes no user parameters; the R-hat and
#'   effective sample size thresholds are the published defaults of
#'   Vehtari and others (2021), stated in `thiessen()`.
#' @srrstatsNA {BS5.4} Only one convergence checker exists, so no
#'   choice-of-checker detail is returned.
#' @srrstatsNA {BS7.3} The package claims no efficiency scaling, and
#'   wall-clock assertions are environment-dependent, so no timing test is
#'   included; statistical behaviour at two sizes is covered by the
#'   calibration suite.
#' @noRd
NULL
