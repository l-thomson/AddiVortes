# Print a fitted model

Print a fitted model

## Usage

``` r
# S3 method for class 'thiessen'
print(x, ...)
```

## Arguments

- x:

  An object of class `"thiessen"`.

- ...:

  Ignored.

## Value

`x`, invisibly.

## Examples

``` r
n <- 60
x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
control <- thiessen_control(
  tessellations = 10,
  general_params = general_params(burn_in = 20, draws = 40)
)
print(thiessen(x, y, control, seed = 1))
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, control = control, seed = 1)
#> gaussian model, 60 observations, 2 covariates
#> 10 tessellations, 40 draws kept after 20 burn-in, thinning 1
#> In-sample RMSE 0.04022, seed 1
#> 1 chain; R-hat and effective sample sizes need two or more chains
```
