# The covariate-inclusion prior of the ensembles

The covariate-inclusion prior of the ensembles

## Usage

``` r
structure_params(omega = NULL)
```

## Arguments

- omega:

  Dimension-count prior parameter omega; omega / p is the prior
  probability of including a covariate. `NULL`, the default, resolves to
  min(3, p) at fit. Must satisfy 0 \< omega \<= p.

## Value

An object of class `"structure_params"`.

## See also

[`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md)

## Examples

``` r
structure_params(omega = 2)
#> structure_params(omega = 2)
```
