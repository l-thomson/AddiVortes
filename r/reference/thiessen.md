# Fit an AddiVortes model

AddiVortes is Bayesian regression on a sum of Voronoi tessellations
(Stone and Gosling, 2025): the mean function is a sum of m
tessellations, each with a mean per cell, drawn by the Gibbs sampler of
the paper. It stands to BART (Chipman, George and McCulloch, 2010) as a
tessellation stands to a tree: a cell is a region of the covariate space
rather than a box, so a boundary oblique to the axes costs one cell
rather than many splits.

## Usage

``` r
thiessen(x, ...)

# Default S3 method
thiessen(x, y, control = thiessen_control(), seed = NULL, chains = 1, ...)

# S3 method for class 'data.frame'
thiessen(x, y, control = thiessen_control(), seed = NULL, chains = 1, ...)

# S3 method for class 'formula'
thiessen(
  formula,
  data,
  control = thiessen_control(),
  seed = NULL,
  chains = 1,
  ...
)
```

## Arguments

- x:

  A numeric matrix of covariates, one row per observation, or a data
  frame. A numeric vector is taken as one column.

- ...:

  Passed to the method.

- y:

  The response: a numeric vector of length `nrow(x)`, or a two-level
  factor. Under the probit model the values must be 0 and 1.

- control:

  An object of class `"thiessen_control"`, from
  [`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md).

- seed:

  The seed of the chain. `NULL`, the default, draws one from R's stream,
  so [`set.seed()`](https://rdrr.io/r/base/Random.html) governs; a whole
  number in `[0, 2^53]` passes to the core unchanged, so the same value
  reproduces the same draws for a given package version and platform.

- chains:

  The number of chains to run, a whole number. Each chain has its own
  seed, derived from `seed` in the core, and the draws of the chains are
  pooled. Two or more chains give the convergence diagnostics; one chain
  does not.

- formula:

  A two-sided formula. The left side names the response and the right
  side the covariates, `.` for every remaining column.

- data:

  A data frame holding the columns the formula names.

## Value

An object of class `"thiessen"`: a list with the fitted state, the
resolved configuration, the number of chains and of kept draws, the
convergence diagnostics where two or more chains ran, the seed used, the
design, the response, the fitted values, the residuals, the hardhat
blueprint where one applies, and the call.

## Details

A factor covariate becomes d - 1 treatment-contrast indicators, the
first level as reference, as `model.matrix` and upstream AddiVortes
encode it. Where `control` declares a `metric`, one entry per column,
factors are passed as integer level codes instead and each factor column
must declare `"categorical"`.

A factor response must have two levels and becomes 0 and 1 with the
first level as the zero, as `glm` treats one. An ordered factor is
encoded as an unordered one; the ordering is not used.

Missing (`NA`) and non-finite values in the covariates or the response
are rejected with an error; no row is dropped silently.

[`stats::update()`](https://rdrr.io/r/stats/update.html) works on a fit:
the call is stored, so `update(fit, seed = 2)` refits with that argument
replaced.

With `chains` of two or more, the chains are run in turn with the seeds
the core derives from `seed`, their draws are pooled, and the fit
carries rank-normalised split R-hat and the bulk and tail effective
sample sizes of sigma and of the mean function at up to twenty training
rows
([`posterior::summarise_draws()`](https://mc-stan.org/posterior/reference/draws_summary.html)).
A fit warns, and [`print()`](https://rdrr.io/r/base/print.html) and
[`summary()`](https://rdrr.io/r/base/summary.html) repeat the warning,
where R-hat exceeds 1.01 or an effective sample size falls below 400
(Vehtari and others, 2021). A fit of one chain says so instead.

## Progress

Progress over the sweep schedule is signalled with progressr, so a
session reports it after
[`progressr::handlers()`](https://progressr.futureverse.org/reference/handlers.html)
and nothing is printed by default; `progressr::handlers(global = TRUE)`
sets one for a whole session. The schedule raises one progression per
sweep, to a maximum of a hundred over the sweeps of every chain. The
draws do not depend on whether a handler is set.

## Persistence

A fit is a plain R object holding the sampler state, so
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) writes one and a
later session reads it and predicts the same values, with no refit. A
fit written by a build with the core's `experimental` feature and read
by a build without it errors with the condition class `thiessen_error`,
naming the feature, at the first call that needs the state.

## Conditions

Errors raised by this package and by the core carry the condition class
`thiessen_error`, and its warnings carry `thiessen_warning`, so either
can be handled or silenced by class rather than by message. The
convergence warning fires on any short schedule, so silencing it
deliberately is a routine need:

    withCallingHandlers(
      fit <- thiessen(x, y, control, chains = 2),
      thiessen_warning = function(condition) {
        invokeRestart("muffleWarning")
      }
    )

## References

Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
Bayesian additive regression trees. *The Annals of Applied Statistics*
4(1), 266-298.
[doi:10.1214/09-AOAS285](https://doi.org/10.1214/09-AOAS285)

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. *Journal of Computational and Graphical
Statistics* 34(3), 859-871.
[doi:10.1080/10618600.2024.2414104](https://doi.org/10.1080/10618600.2024.2414104)

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Buerkner, P.-C.
(2021). Rank-normalization, folding, and localization: an improved R-hat
for assessing convergence of MCMC. *Bayesian Analysis* 16(2), 667-718.
[doi:10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221)

## Examples

``` r
n <- 60
x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
control <- thiessen_control(
  tessellations = 10,
  general_params = general_params(burn_in = 20, draws = 40)
)
fit <- thiessen(x, y, control, seed = 1)
fit
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, control = control, seed = 1)
#> gaussian model, 60 observations, 2 covariates
#> 10 tessellations, 40 draws kept after 20 burn-in, thinning 1
#> In-sample RMSE 0.04022, seed 1
#> 1 chain; R-hat and effective sample sizes need two or more chains

# Two chains add the convergence diagnostics; one chain reports none.
thiessen(x, y, control, seed = 1, chains = 2)
#> Warning: The chains may not have converged: largest R-hat 1.739 (threshold 1.01), smallest effective sample size 4 (threshold 400). Run more draws or more chains.
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, control = control, seed = 1, chains = 2)
#> gaussian model, 60 observations, 2 covariates
#> 10 tessellations, 80 draws kept after 20 burn-in, thinning 1
#> In-sample RMSE 0.03779, seed 1
#> 2 chains, largest R-hat 1.739, smallest effective sample size 4
#> Warning: The chains may not have converged: largest R-hat 1.739 (threshold 1.01), smallest effective sample size 4 (threshold 400). Run more draws or more chains.

frame <- data.frame(y = y, a = x[, 1], b = factor(x[, 2] > 0))
thiessen(y ~ a + b, frame, control, seed = 1)
#> AddiVortes fit
#> Call: thiessen(formula = y ~ a + b, data = frame, control = control,      seed = 1)
#> gaussian model, 60 observations, 2 covariates
#> 10 tessellations, 40 draws kept after 20 burn-in, thinning 1
#> In-sample RMSE 0.04022, seed 1
#> 1 chain; R-hat and effective sample sizes need two or more chains
```
