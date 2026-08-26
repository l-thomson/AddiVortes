
<!-- README.md is generated from README.Rmd. Please edit that file -->

# thiessen

Bayesian regression on a sum of Voronoi tessellations (AddiVortes; Stone
and Gosling, 2025, <doi:10.1080/10618600.2024.2414104>), a variant of
BART (Chipman, George and McCulloch, 2010) in which a cell is a region
of the covariate space rather than a box. The package provides the
Gaussian model of the paper together with its published variants, Binary
AddiVortes (probit classification) and H-AddiVortes (heteroscedastic
variance).

The sampler is the `thiessen` Rust crate, built from sources vendored in
the package. The method and all credit for it belong to its authors;
this package is an independent implementation, and its test suite
compares posterior summaries against the authors' R package,
[AddiVortes](https://github.com/johnpaulgosling/AddiVortes).

## Installation

The package is not yet on CRAN, and it compiles Rust, so the build needs
a toolchain besides R's own:

- rustc 1.74 or later with Cargo, from [rustup](https://rustup.rs).
- On Windows, [Rtools](https://cran.r-project.org/bin/windows/Rtools/)
  as well, and the Rust target the R build links against. Under the
  usual Rtools that is `x86_64-pc-windows-gnu`, so
  `rustup target add x86_64-pc-windows-gnu`; `configure.win` prints the
  target it selected, which differs on the clang and ARM builds.

Install the toolchain before starting R, or restart the session
afterwards: rustup puts `cargo` on the path of new sessions only, and a
session older than the install fails the build as though no toolchain
were there. Every crate the build needs ships with the package, so the
compilation itself uses no network.

``` r
install.packages("remotes")
remotes::install_github("l-thomson/thiessen", subdir = "r")
```

The articles are on the website named below, so the install leaves them
out. `build_vignettes = TRUE` installs them locally instead, and needs
`knitr` and `rmarkdown` present to build them.

## Example

``` r
library(thiessen)

set.seed(1)
n <- 200
x <- cbind(runif(n), runif(n))
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2] + rnorm(n, sd = 0.1)

fit <- thiessen(x, y, seed = 1)
fit
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, seed = 1)
#> gaussian model, 200 observations, 2 covariates
#> 200 tessellations, 4000 draws kept after 200 burn-in, thinning 1
#> In-sample RMSE 0.07966, seed 1
#> 4 chains, largest R-hat 1.008, smallest effective sample size 899
```

``` r
head(predict(fit, x))
#> [1] 0.2161658 0.1520645 0.3659401 0.4203385 0.2189898 0.6780119
sigma(fit)
#> [1] 0.09372794
```

`plot(fit)` traces the per-draw sampler diagnostics, one panel each. It
is a convergence check rather than a display of the fit; for the fit
itself pass `posterior::as_draws_df(fit)` to bayesplot.

## Documentation

- `vignette("thiessen")`, getting started: a fit, its methods, the
  convergence diagnostics and what to do when they warn.
- `vignette("binary-addivortes")`, probit classification.
- `vignette("h-addivortes")`, the heteroscedastic model, where the
  spread varies with x.
- `vignette("posterior")`, the draws themselves: posterior, loo,
  bayesplot and tidybayes.
- `vignette("control-surface")`, every configuration group and the
  defaults the core resolves.
- `vignette("sampler-api")`, driving the Gibbs loop from R to build a
  model the package does not ship.

The same articles are rendered with their output, and the reference
pages beside them, on the [package
website](https://l-thomson.github.io/thiessen/r/).

Read them in that order: getting started, then the article for the model
in hand, then the control surface as a reference, and the sampler API
only when extending the method.
[`docs/models.md`](https://github.com/l-thomson/thiessen/blob/dev/docs/models.md)
gives the correspondence between this package's parameters, the paper's,
and those of CRAN AddiVortes and the BART family.

## Working with a fit

A formula interface takes data frames, with factor covariates encoded as
`model.matrix` encodes them, and the default four `chains` carry R-hat
and effective sample size diagnostics.

`posterior::as_draws_df()`, `summarise_draws()`, `posterior_predict()`,
`posterior_epred()`, `log_lik()` and `predictive_interval()` work on a
fit, so `loo::loo()`, bayesplot and tidybayes take one without
adaptation. `vignette("posterior")` runs each of them.

`saveRDS()` writes a fit and a later session reads it and predicts the
same values; the sampler state travels in the object.

Progress over the whole fit is signalled with progressr, so nothing is
printed until a session chooses a handler:

``` r
progressr::handlers(global = TRUE)
progressr::handlers("txtprogressbar")
fit <- thiessen(x, y, seed = 1)
```

The sweeps, pooling the draws and the convergence summary each report,
so the bar stands until the fit is complete. Reporting does not change
the draws.

## Covariate scaling

Pass raw data. Inside the sampler a Euclidean column is min-max scaled
to \[-0.5, 0.5\] over its training range, which is what makes one
distance comparable across columns; a categorical column and a spherical
one are not scaled. The centre-proposal scale `sigma_c` lives on that
internal scale, so `sigma_c = 1` is the full training range of a
Euclidean column.

## Runtime

Cost is linear in the sweeps and close to linear in the rows and in the
tessellation count. The default schedule is 1200 sweeps, 200 of burn-in
and 1000 draws, and a default fit runs four chains of it on
`getOption("mc.cores", 1L)` threads, so a session that sets nothing pays
four chains on one core. As one calibration point, 600 sweeps of n =
1000 and p = 3 with 100 tessellations take about 7 seconds on one core
of a 2025 laptop. `options(mc.cores = 4)` runs the chains on four cores
for the same draws; the fit then takes less than half the one-thread
time (about 45 per cent at n = 200 on four cores of a 2025 laptop), so
it costs under two chains rather than four.

A default fit is short for the diagnostics it carries: on Friedman \#1
with n = 200 and p = 10 the smallest effective sample size over the
monitored points is about 100 (threshold 400) and the largest R-hat
about 1.05 (threshold 1.01), so the fit warns. More draws per chain,
`general_params(draws = )`, is the answer; `thinning` is not.

## Priors

The priors are those of Stone and Gosling (2025) and are set through
`thiessen_control()`. The outcome family carries the prior on the noise:
`gaussian_outcome(nu, q)` scales the inverse chi-squared prior on
sigma^2 so that a proportion `q` of its mass lies below the sample
variance of the response. The mean ensemble's priors sit in
`term_params()`: `k` fixes the spread of the cell-value prior,
`lambda_c` is the Poisson rate of the cell-count prior, and the nested
`geometry_params()` and `structure_params()` hold the centre-proposal
scale `sigma_c` and the covariate-inclusion weight `omega`. Setting
`prior_only = TRUE` in `general_params()` draws from the prior alone.
`vignette("control-surface")` walks the whole surface.

## Scope

This release carries the Gaussian, probit and heteroscedastic models.
The core crate also holds tobit, accelerated-failure-time,
interval-censored and ordinal outcome models behind its `experimental`
Cargo feature, which this package enables in no build: `tobit()` is not
a function here, and `thiessen_control()` accepts the Gaussian and
probit families only. `core_version()` and `core_experimental()` report
the core version and the feature setting of the build in use, and a bug
report should carry both. Each of the four graduates on its own once
calibrated, with no build flag to turn on, under the policy in
[`docs/experimental.md`](https://github.com/l-thomson/thiessen/blob/dev/docs/experimental.md).
Until then any of them can be built in R against the sampler.

## Building a model the package does not ship

A new outcome family, a censoring scheme or an imputation scheme is
written in R against the sampler, with no Rust and no recompilation.
`thiessen_sampler()` hands over the Gibbs loop one sweep at a time and
allows the response to be rewritten between sweeps, which is what a
latent-Gaussian augmentation needs. `vignette("sampler-api")`
reimplements the probit family in fifteen lines of R and checks it
against the built-in one.

## References

Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
Bayesian additive regression trees. The Annals of Applied Statistics
4(1), 266-298. <doi:10.1214/09-AOAS285>

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. Journal of Computational and Graphical Statistics
34(3), 859-871. <doi:10.1080/10618600.2024.2414104>
