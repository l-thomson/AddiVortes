# Posterior predictions from a fitted model

Posterior predictions from a fitted model

## Usage

``` r
# S3 method for class 'thiessen'
predict(
  object,
  newdata = NULL,
  type = c("mean", "draws", "latent", "variance", "probs"),
  interval = c("none", "credible", "prediction"),
  level = 0.95,
  threads = NULL,
  ...
)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- newdata:

  New covariates. A fit from a formula or a data frame takes a data
  frame, whose columns are matched to the fitted design by name and
  type; a fit from a matrix takes a numeric matrix with the fitted
  columns. `NULL`, the default, predicts at the training rows.

- type:

  The quantity: `"mean"`, the posterior mean of the response (the
  probability under the probit model, the expected category under the
  ordinal model, f(x) on the log-time scale under the AFT model);
  `"draws"`, that quantity for every kept draw; `"latent"`, the mean
  function f for every kept draw; `"variance"`, the variance of y given
  f for every kept draw; `"probs"`, under the ordinal model only, the
  posterior-mean probability of each category, the
  [`MASS::polr`](https://rdrr.io/pkg/MASS/man/polr.html) name.

- interval:

  `"none"`, the default; `"credible"` for the interval of the posterior
  mean; `"prediction"` for the posterior predictive interval. Only with
  `type = "mean"`.

- level:

  The mass of a central interval. Default 0.95.

- threads:

  The number of threads, a whole number; `NULL`, the default, is the
  count the fit was made with.

- ...:

  Ignored.

## Value

For `type = "mean"` and `interval = "none"`, a numeric vector of length
`nrow(newdata)`; with an interval, a matrix of that many rows with
columns `fit`, `lower` and `upper`. For `type = "probs"`, a matrix of
one row per row of `newdata` and one column per category, named by the
levels of the response. For the other types, a matrix of one row per
kept draw and one column per row of `newdata`.

## Details

The rows of `newdata` are split over `threads` threads, each chunk
evaluated on a thread of its own; the values do not depend on the count.

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
head(predict(fit))
#> [1] 0.3862057 0.6317637 0.3862057 0.6317637 0.3846993 0.5913465
head(predict(fit, interval = "credible"))
#>            fit     lower     upper
#> [1,] 0.3862057 0.3282718 0.4422589
#> [2,] 0.6317637 0.5499637 0.7089330
#> [3,] 0.3862057 0.3282718 0.4422589
#> [4,] 0.6317637 0.5499637 0.7089330
#> [5,] 0.3846993 0.3282718 0.4422589
#> [6,] 0.5913465 0.5392440 0.6456429
```
