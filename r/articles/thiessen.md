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
#> Warning in thiessen(y ~ ., d, control, seed = 1): The chains may not have
#> converged: largest R-hat 1.457 (threshold 1.01), smallest effective sample size
#> 8 (threshold 400). Run more draws or more chains.
fit
#> AddiVortes fit
#> Call: thiessen(formula = y ~ ., data = d, control = control, seed = 1)
#> gaussian model, 200 observations, 4 covariates
#> 25 tessellations, 800 draws kept after 100 burn-in, thinning 1
#> In-sample RMSE 0.07421, seed 1
#> 4 chains, largest R-hat 1.457, smallest effective sample size 8
#> Warning: The chains may not have converged: largest R-hat 1.457 (threshold 1.01), smallest effective sample size 8 (threshold 400). Run more draws or more chains.
```

A factor covariate becomes d - 1 treatment-contrast indicators, the
first level as reference, as `model.matrix` encodes one.
[`predict()`](https://rdrr.io/r/stats/predict.html) returns the
posterior mean and matches new data by column name and type:

``` r

head(predict(fit, d))
#> [1] 0.2487482 0.1422802 0.6507389 0.8504028 0.4615400 0.9697501
head(predict(fit, d, interval = "credible", level = 0.9))
#>            fit      lower     upper
#> [1,] 0.2487482 0.16645838 0.3262066
#> [2,] 0.1422802 0.06682134 0.2203331
#> [3,] 0.6507389 0.55390796 0.7514768
#> [4,] 0.8504028 0.75976749 0.9483209
#> [5,] 0.4615400 0.37225783 0.5424575
#> [6,] 0.9697501 0.88757243 1.0543148
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
#> 200.00000000   0.09917727   0.07421256
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
#> Warning in thiessen(y ~ ., d, control, seed = 1): The chains may not have
#> converged: largest R-hat 1.457 (threshold 1.01), smallest effective sample size
#> 8 (threshold 400). Run more draws or more chains.
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
The default four `chains` carry rank-normalised split R-hat and
effective sample sizes, and a fit warns where they cross the usual
thresholds, as every fit in this article does at its short schedule. The
chains run on `getOption("mc.cores", 1L)` threads, so
`options(mc.cores = 4)` runs them on four cores for the same draws.

``` r

summary(fit)
#> AddiVortes fit
#> Call: thiessen(formula = y ~ ., data = d, control = control, seed = 1)
#> gaussian model, 200 observations, 4 covariates
#> 25 tessellations, 800 draws kept after 100 burn-in, thinning 1
#> 
#> Residuals:
#>         2.5%          25%          50%          75%        97.5% 
#> -0.158475949 -0.048499558  0.001712749  0.045790803  0.131873618 
#> 
#> sigma:
#>       2.5%        25%        50%        75%      97.5% 
#> 0.08810571 0.09460426 0.09903118 0.10307944 0.11194470 
#> 
#> In-sample RMSE 0.07421
#> 4 chains, largest R-hat 1.457, smallest effective sample size 8
#> Warning: The chains may not have converged: largest R-hat 1.457 (threshold 1.01), smallest effective sample size 8 (threshold 400). Run more draws or more chains.
```

[`variable_inclusion()`](https://l-thomson.github.io/thiessen/r/reference/variable_inclusion.md)
reports the share of the ensemble’s active dimensions falling on each
covariate.

``` r

variable_inclusion(fit)
#>         a         b      glow      gmid 
#> 0.2896471 0.2531981 0.2334469 0.2237079
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
draws per chain, `general_params(draws = )`, is the first answer, then
more chains; `thinning` reduces the stored draws without adding
information, so it is not the remedy for a low effective sample size.
The default schedule is itself short for the thresholds: on Friedman \#1
with n = 200 and p = 10 a default fit reaches a smallest effective
sample size of about 100 against the threshold of 400 and a largest
R-hat of about 1.05 against 1.01, so it warns. The warning fires on any
short schedule, this article’s included, and carries the condition class
`thiessen_warning`, so it can be silenced by class rather than by
message:

``` r

withCallingHandlers(
  quiet <- thiessen(y ~ ., d, control, seed = 1),
  thiessen_warning = function(condition) {
    invokeRestart("muffleWarning")
  }
)
quiet$n_chains
#> [1] 4
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
