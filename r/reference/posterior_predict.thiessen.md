# Draw from the posterior predictive distribution

One replicate per kept draw from the fitted family's observation model:
the mean function of that draw plus a Gaussian residual at the scale of
that draw under the Gaussian and heteroscedastic models; Bernoulli
labels under the probit model; category codes, 0 to K - 1, from the
latent value against the cutpoints of that draw under the ordinal model;
a time, the exponential of the log-time draw, under the AFT model; the
value clipped to the censoring limits under the tobit model; the working
value under the interval-censored model; and a Student-t or Laplace
error at the drawn scale under those models. The residuals are drawn
from R's stream, so [set.seed()](https://rdrr.io/r/base/Random.html)
governs them; they are not part of the chain the core draws.

## Usage

``` r
# S3 method for class 'thiessen'
posterior_predict(object, newdata = NULL, ...)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- newdata:

  New covariates, as
  [`predict.thiessen()`](https://l-thomson.github.io/thiessen/r/reference/predict.thiessen.md)
  takes them. `NULL`, the default, is the training rows.

- ...:

  Ignored.

## Value

A double matrix, one row per kept draw and one column per row of
`newdata`.

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
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
dim(posterior_predict(fit))
#> [1] 160  60
```
