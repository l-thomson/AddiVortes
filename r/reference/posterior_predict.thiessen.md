# Draw from the posterior predictive distribution

One replicate per kept draw: the mean function of that draw plus a
residual under the model, Bernoulli labels under the probit model. The
residuals are drawn from R's stream, so
[`set.seed()`](https://rdrr.io/r/base/Random.html) governs them; they
are not part of the chain the core draws.

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
dim(posterior_predict(fit))
#> [1] 40 60
```
