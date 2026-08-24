# Residual standard deviation of a fitted model

Residual standard deviation of a fitted model

## Usage

``` r
# S3 method for class 'thiessen'
sigma(object, ...)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- ...:

  Ignored.

## Value

A single number: the posterior mean of sigma under the Gaussian model,
and 1 under the probit model, whose latent scale is fixed. The
heteroscedastic model has no single residual scale; use
`predict(type = "variance")`.

## Examples

``` r
n <- 60
x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
control <- thiessen_control(
  tessellations = 10,
  general_params = general_params(burn_in = 20, draws = 40)
)
sigma(thiessen(x, y, control, seed = 1))
#> [1] 0.05876912
```
