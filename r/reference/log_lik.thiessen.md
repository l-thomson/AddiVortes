# Pointwise log-likelihood of a fitted model

The matrix [`loo::loo()`](https://mc-stan.org/loo/reference/loo.html)
takes.

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

  The response at `newdata`. Taken from `newdata` where the fit came
  from a formula and `newdata` carries the response column.

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
#> Warning: Some Pareto k diagnostic values are too high. See help('pareto-k-diagnostic') for details.
#> 
#> Computed from 40 by 60 log-likelihood matrix.
#> 
#>          Estimate  SE
#> elpd_loo     87.7 3.6
#> p_loo        11.2 1.8
#> looic      -175.4 7.3
#> ------
#> MCSE of elpd_loo is NA.
#> MCSE and ESS estimates assume independent draws (r_eff=1).
#> 
#> Pareto k diagnostic values:
#>                           Count Pct.    Min. ESS
#> (-Inf, 0.38]   (good)     33    55.0%   12      
#>    (0.38, 1]   (bad)      26    43.3%   <NA>    
#>     (1, Inf)   (very bad)  1     1.7%   <NA>    
#> See help('pareto-k-diagnostic') for details.
```
