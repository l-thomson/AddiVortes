# Pointwise log-likelihood of a fitted model

The matrix [`loo::loo()`](https://rdrr.io/pkg/loo/man/loo.html) takes.

## Usage

``` r
# S3 method for class 'thiessen'
log_lik(object, newdata = NULL, y = NULL, ...)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- newdata:

  New covariates, as
  [`predict.thiessen()`](https://l-thomson.github.io/thiessen/r/reference/predict.thiessen.md)
  takes them. `NULL`, the default, is the training rows.

- y:

  The response at `newdata`, in the shape
  [`thiessen()`](https://l-thomson.github.io/thiessen/r/reference/thiessen.md)
  took: a `Surv` under the AFT and interval-censored families. Taken
  from `newdata` where the fit came from a formula and `newdata` carries
  the response column.

- ...:

  Ignored.

## Value

A double matrix, one row per kept draw and one column per observation.

## Examples

``` r
if (requireNamespace("loo", quietly = TRUE)) {
  n <- 60
  x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
  y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
  control <- thiessen_control(
    tessellations = 10,
    general_params = general_params(burn_in = 20, draws = 40)
  )
  fit <- thiessen(x, y, control, seed = 1)
  loo::loo(log_lik(fit))
}
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
#> Warning: Some Pareto k diagnostic values are too high. See help('pareto-k-diagnostic') for details.
#> 
#> Computed from 160 by 60 log-likelihood matrix.
#> 
#>          Estimate  SE
#> elpd_loo     81.3 4.1
#> p_loo        17.4 2.8
#> looic      -162.5 8.3
#> ------
#> MCSE of elpd_loo is NA.
#> MCSE and ESS estimates assume independent draws (r_eff=1).
#> 
#> Pareto k diagnostic values:
#>                           Count Pct.    Min. ESS
#> (-Inf, 0.55]   (good)     47    78.3%   26      
#>    (0.55, 1]   (bad)      13    21.7%   <NA>    
#>     (1, Inf)   (very bad)  0     0.0%   <NA>    
#> See help('pareto-k-diagnostic') for details.
```
