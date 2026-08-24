# The Gaussian outcome family

The Gaussian observation model of Stone and Gosling (2025): one sigma^2
drawn per sweep. Attaching a variance ensemble
(`variance_params = term_params(tessellations = ...)` with a positive
count) makes the model heteroscedastic, so the residual variance varies
with x.

## Usage

``` r
gaussian_outcome(nu = 6, q = 0.85)
```

## Arguments

- nu:

  Degrees of freedom nu of the sigma^2 prior, sigma^2 ~ nu lambda /
  chi^2_nu. Default 6. A variance ensemble requires nu \> 2.

- q:

  Calibration quantile q of the sigma^2 prior, Pr(sigma \< sigma_hat)
  = q. Default 0.85.

## Value

An object of class `c("thiessen_gaussian", "thiessen_outcome")`.

## See also

[`probit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/probit_outcome.md),
[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md)

## Examples

``` r
gaussian_outcome(nu = 3)
#> gaussian_outcome(nu = 3, q = 0.85)
```
