# Central posterior predictive interval

Central posterior predictive interval

## Usage

``` r
# S3 method for class 'thiessen'
predictive_interval(object, prob = 0.9, newdata = NULL, ...)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- prob:

  The mass of the interval. Default 0.9.

- newdata:

  New covariates, as
  [`predict.thiessen()`](https://l-thomson.github.io/thiessen/r/reference/predict.thiessen.md)
  takes them. `NULL`, the default, is the training rows.

- ...:

  Ignored.

## Value

A double matrix of one row per row of `newdata`, with the lower and
upper bounds as columns named by their percentiles.

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
head(predictive_interval(fit))
#>             5%       95%
#> [1,] 0.2686235 0.4926991
#> [2,] 0.4942976 0.7458845
#> [3,] 0.2686235 0.4926991
#> [4,] 0.4942976 0.7458845
#> [5,] 0.2654548 0.4915973
#> [6,] 0.4784210 0.6966777
```
