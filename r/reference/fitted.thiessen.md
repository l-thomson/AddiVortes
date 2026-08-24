# Fitted values, residuals and observation count of a fitted model

Fitted values, residuals and observation count of a fitted model

## Usage

``` r
# S3 method for class 'thiessen'
fitted(object, ...)

# S3 method for class 'thiessen'
residuals(object, ...)

# S3 method for class 'thiessen'
nobs(object, ...)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- ...:

  Ignored.

## Value

For [`fitted()`](https://rdrr.io/r/stats/fitted.values.html) and
[`residuals()`](https://rdrr.io/r/stats/residuals.html), a numeric
vector of length `nobs(object)`: the posterior mean of the response at
each training row, and the response less that mean. For
[`nobs()`](https://rdrr.io/r/stats/nobs.html), the number of
observations.

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
nobs(fit)
#> [1] 60
head(residuals(fit))
#> [1]  0.113794337  0.084912549  0.048295917  0.021712320 -0.011099737
#> [6]  0.003525624
```
