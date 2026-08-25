# Getting started

AddiVortes is Bayesian regression on a sum of Voronoi tessellations
(Stone and Gosling, 2025, *Journal of Computational and Graphical
Statistics* 34(3), 859-871). It stands to BART as a tessellation stands
to a tree: a cell is a region of the covariate space rather than a box,
so a boundary oblique to the axes costs one cell rather than many
splits. This vignette fits the Gaussian model with the defaults; the
other published variants have their own vignettes, by their paper names.

## A first fit

[`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
takes a formula and a data frame, a data frame and a response, or a
numeric matrix and a response. The sweep schedule here is short so the
vignette builds quickly; the defaults are 200 burn-in sweeps and 1000
kept draws.

``` r

library(thiessen)

set.seed(1)
n <- 200
d <- data.frame(
  a = runif(n),
  b = runif(n),
  g = factor(sample(c("low", "mid", "high"), n, replace = TRUE))
)
d$y <- 3 * (d$a - 0.4)^2 + 0.5 * d$b + 0.3 * (d$g == "high") +
  rnorm(n, sd = 0.1)

control <- thiessen_control(
  tessellations = 25,
  general_params = general_params(burn_in = 100, draws = 200)
)
fit <- thiessen(y ~ ., d, control, seed = 1)
fit
#> AddiVortes fit
#> Call: thiessen(formula = y ~ ., data = d, control = control, seed = 1)
#> gaussian model, 200 observations, 4 covariates
#> 25 tessellations, 200 draws kept after 100 burn-in, thinning 1
#> In-sample RMSE 0.0794, seed 1
#> 1 chain; R-hat and effective sample sizes need two or more chains
```

A factor covariate becomes d - 1 treatment-contrast indicators, the
first level as reference, as `model.matrix` encodes one.
[`predict()`](https://rdrr.io/r/stats/predict.html) returns the
posterior mean and matches new data by column name and type:

``` r

head(predict(fit, d))
#> [1] 0.2373794 0.1139564 0.6631289 0.8431266 0.4777981 0.9805984
head(predict(fit, d, interval = "credible", level = 0.9))
#>            fit      lower     upper
#> [1,] 0.2373794 0.14127244 0.3191082
#> [2,] 0.1139564 0.03866678 0.1859433
#> [3,] 0.6631289 0.56411400 0.7900226
#> [4,] 0.8431266 0.75900457 0.9378979
#> [5,] 0.4777981 0.41383460 0.5433476
#> [6,] 0.9805984 0.91549249 1.0467040
```

`predict(interval = "prediction")` covers a new observation rather than
the mean, and `type = "draws"` returns the per-draw values.

[`fitted()`](https://rdrr.io/r/stats/fitted.values.html),
[`residuals()`](https://rdrr.io/r/stats/residuals.html),
[`sigma()`](https://rdrr.io/r/stats/sigma.html),
[`nobs()`](https://rdrr.io/r/stats/nobs.html),
[`print()`](https://rdrr.io/r/base/print.html),
[`summary()`](https://rdrr.io/r/base/summary.html) and
[`plot()`](https://rdrr.io/r/graphics/plot.default.html) behave as they
do on any model object, and `update(fit, seed = 2)` refits with that one
argument replaced.

``` r

c(nobs = nobs(fit), sigma = sigma(fit),
  rmse = sqrt(mean(residuals(fit)^2)))
#>         nobs        sigma         rmse 
#> 200.00000000   0.09912894   0.07940401
```

## Covariate scaling

Pass raw data. Inside the sampler a Euclidean column is min-max scaled
to \[-0.5, 0.5\] over its training range, which is what makes one
distance comparable across columns of different units; a categorical
column and a spherical one are not scaled. The centre-proposal scale
`sigma_c` is on that internal scale, so `sigma_c = 1` is the full
training range of a Euclidean column. Nothing else about a fit depends
on the units the covariates arrive in.

## The configuration

The configuration has four parts, named as the core stores them: an
outcome family from
[`gaussian_outcome()`](https://l-thomson.github.io/thiessen/r/reference/gaussian_outcome.md)
or
[`probit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/probit_outcome.md),
one
[`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md)
group per ensemble, and the sweep schedule from
[`general_params()`](https://l-thomson.github.io/thiessen/r/reference/general_params.md).
`thiessen_control(tessellations = )` is the one shortcut, setting the
mean ensemble’s size. The control-surface vignette walks through every
group.

``` r

thiessen_control(
  outcome = gaussian_outcome(nu = 6, q = 0.85),
  mean_params = term_params(tessellations = 200, k = 3, lambda_c = 5)
)
#> <thiessen_control>
#>   outcome         gaussian_outcome(nu = 6, q = 0.85)
#>   mean_params     term_params(tessellations = 200, k = 3, lambda_c = 5)
#>   variance_params none (constant spread)
#>   general_params  general_params(burn_in = 200, draws = 1000, thinning = 1, prior_only = FALSE)
```

## Reproducibility

`seed = NULL`, the default, draws the chain’s seed from R’s stream, so
[`set.seed()`](https://rdrr.io/r/base/Random.html) governs; a whole
number reproduces the same draws for a given package version and
platform. The resolved seed is on the fit.

``` r

again <- thiessen(y ~ ., d, control, seed = 1)
identical(predict(fit, d), predict(again, d))
#> [1] TRUE
```

## Saving a fit

A fit is a plain R object holding the sampler state, so
[`saveRDS()`](https://rdrr.io/r/base/readRDS.html) writes one and a
later session reads it and predicts the same values without a refit.

``` r

path <- tempfile(fileext = ".rds")
saveRDS(fit, path)
identical(predict(readRDS(path), d), predict(fit, d))
#> [1] TRUE
```

## Diagnostics

The established generics are available:
[`posterior::as_draws_df()`](https://mc-stan.org/posterior/reference/draws_df.html),
[`summary()`](https://rdrr.io/r/base/summary.html), and the rstantools
generics
[`posterior_predict()`](https://mc-stan.org/rstantools/reference/posterior_predict.html),
[`posterior_epred()`](https://mc-stan.org/rstantools/reference/posterior_epred.html),
[`log_lik()`](https://mc-stan.org/rstantools/reference/log_lik.html) and
[`predictive_interval()`](https://mc-stan.org/rstantools/reference/predictive_interval.html).
Two or more `chains` add rank-normalised split R-hat and effective
sample sizes, and a fit warns where they cross the usual thresholds.

``` r

summary(fit)
#> AddiVortes fit
#> Call: thiessen(formula = y ~ ., data = d, control = control, seed = 1)
#> gaussian model, 200 observations, 4 covariates
#> 25 tessellations, 200 draws kept after 100 burn-in, thinning 1
#> 
#> Residuals:
#>         2.5%          25%          50%          75%        97.5% 
#> -0.163260078 -0.049799348  0.002716381  0.055474697  0.145138811 
#> 
#> sigma:
#>       2.5%        25%        50%        75%      97.5% 
#> 0.08921890 0.09456918 0.09861395 0.10281368 0.11096115 
#> 
#> In-sample RMSE 0.0794
#> 1 chain; R-hat and effective sample sizes need two or more chains
```

[`variable_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/variable_inclusion.md)
reports the share of the ensemble’s active dimensions falling on each
covariate.

``` r

variable_inclusion(fit)
#>         a         b      glow      gmid 
#> 0.2816334 0.2546651 0.2512142 0.2124872
```

Read that as where the ensemble spent its dimensions, not as variable
selection. At the default `omega` of `min(3, p)` every dimension is
always active when p is 3 or fewer, so the proportions are then uniform
by construction; p is 4 here after the factor encoding, and the
informative covariates separate from the rest only weakly.

[`thiessen_diagnostics()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_diagnostics.md)
returns the per-draw quantities as a data frame, for
[`bayesplot::mcmc_trace()`](https://mc-stan.org/bayesplot/reference/MCMC-traces.html)
or a plot of your own. It is not for printing: a bare
[`print()`](https://rdrr.io/r/base/print.html) dumps every draw.

## Progress

Progress over the whole fit is signalled with progressr, and nothing is
printed by default: the package raises the conditions and the session
decides whether and how to report them. Choose a handler once and it
applies to every fit afterwards.

``` r

progressr::handlers(global = TRUE)
progressr::handlers("txtprogressbar")
fit <- thiessen(y ~ ., d, control, seed = 1)
```

[`progressr::with_progress()`](https://progressr.futureverse.org/reference/with_progress.html)
scopes a handler to one expression instead. The chunks here are not
evaluated because progressr renders nothing into a static document.

The schedule raises one progression per sweep, to a maximum of a hundred
over the sweeps of every chain, then pooling the draws and the
convergence summary. Both run after the last sweep, and pooling predicts
at every training row for every kept draw, so on a long schedule over
many rows it costs about twice what the sweeps cost. It therefore
carries their weight in the bar rather than a step of it: the bar is
around a third of the way along when the sweeps end, and it stands until
the fit is complete. Pooling is one call into the core, so the bar rests
there rather than advancing through it. Each phase names itself in a
sticky message, which a terminal handler pushes above the bar rather
than overwriting, so the phase that is running is named whatever handler
is set. Reporting does not change the draws.

## Troubleshooting

### The fit prints nothing while it runs

That is the default, not a hang. See Progress above.

### R-hat or an effective sample size warns

[`summary()`](https://rdrr.io/r/base/summary.html) reports both. More
draws is the first answer, then more chains; `thinning` reduces the
stored draws without adding information, so it is not the remedy for a
low effective sample size. The warning fires on any short schedule, this
article’s included, and carries the condition class `thiessen_warning`,
so it can be silenced by class rather than by message:

``` r

withCallingHandlers(
  two <- thiessen(y ~ ., d, control, chains = 2, seed = 1),
  thiessen_warning = function(condition) {
    invokeRestart("muffleWarning")
  }
)
two$n_chains
#> [1] 2
```

### A saved fit will not load

A fit written by a build carrying the core’s experimental feature and
read by a build without it errors with the class `thiessen_error`,
naming the feature.
[`core_version()`](https://l-thomson.github.io/thiessen/r/reference/core_version.md)
and
[`core_experimental()`](https://l-thomson.github.io/thiessen/r/reference/core_experimental.md)
report the core version and the feature setting of the build in use; a
bug report should carry both.

``` r

c(core = core_version(), experimental = core_experimental())
#>         core experimental 
#>      "0.3.0"      "FALSE"
```
