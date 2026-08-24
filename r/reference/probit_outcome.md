# The binary probit outcome family

P(y = 1 \| x) = Phi(c + f(x)) with the Albert and Chib (1993) latent
augmentation. The latent scale is fixed at 1 for identification, so a
variance ensemble is not available under this family.

## Usage

``` r
probit_outcome(offset = NULL)
```

## Arguments

- offset:

  The offset c. `NULL`, the default, resolves to Phi^-1(ybar) at fit,
  the BART `binaryOffset` default.

## Value

An object of class `c("thiessen_probit", "thiessen_outcome")`.

## See also

[`gaussian_outcome()`](https://l-thomson.github.io/thiessen/r/reference/gaussian_outcome.md),
[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md)

## Examples

``` r
probit_outcome()
#> probit_outcome()
```
