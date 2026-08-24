# Summarise a fitted model

Summarise a fitted model

## Usage

``` r
# S3 method for class 'thiessen'
summary(object, ...)

# S3 method for class 'summary.thiessen'
print(x, ...)
```

## Arguments

- object:

  An object of class `"thiessen"`.

- ...:

  Ignored.

- x:

  An object of class `"summary.thiessen"`.

## Value

An object of class `"summary.thiessen"`: a list of the model, the
dimensions of the fit, the sweep schedule, the in-sample root mean
squared error, the quantiles of the residuals, the quantiles of the
posterior draws of sigma where the model has one, and the convergence
diagnostics where two or more chains ran.

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
summary(thiessen(x, y, control, seed = 1))
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, control = control, seed = 1)
#> gaussian model, 60 observations, 2 covariates
#> 10 tessellations, 40 draws kept after 20 burn-in, thinning 1
#> 
#> Residuals:
#>         2.5%          25%          50%          75%        97.5% 
#> -0.053658678 -0.023596686 -0.005973156  0.024384175  0.087277594 
#> 
#> sigma:
#>       2.5%        25%        50%        75%      97.5% 
#> 0.04593405 0.04851163 0.05469341 0.07051679 0.07995568 
#> 
#> In-sample RMSE 0.04022
#> 1 chain; R-hat and effective sample sizes need two or more chains
```
