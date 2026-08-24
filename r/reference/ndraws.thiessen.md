# Number of draws and chains of a fitted model

Number of draws and chains of a fitted model

## Usage

``` r
# S3 method for class 'thiessen'
ndraws(x)

# S3 method for class 'thiessen'
nchains(x)
```

## Arguments

- x:

  An object of class `"thiessen"`.

## Value

The number of kept draws over every chain, and the number of chains the
fit ran.

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
posterior::ndraws(fit)
#> [1] 40
```
