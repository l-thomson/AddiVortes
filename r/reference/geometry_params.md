# The covariate space of the ensembles

The covariate space of the ensembles

## Usage

``` r
geometry_params(metric = NULL, sigma_c = 0.8)
```

## Arguments

- metric:

  The metric of each covariate column, in column order: a list whose
  entries are `"euclidean"`, `"categorical"`, or
  `list(spherical = list(sphere = i))` for one coordinate of the sphere
  labelled `i`, its latitudes first and its longitude last, in radians.
  `NULL`, the default, is Euclidean on every column. Entries are matched
  case-sensitively, so `"Euclidean"` is rejected. Non-Euclidean columns
  are not scaled.

- sigma_c:

  Prior and proposal standard deviation sigma_c of a centre coordinate.
  A Euclidean column is min-max scaled to \[-0.5, 0.5\] over its
  training range inside the sampler and `sigma_c` is on that scale, so 1
  is the full range of a column. Default 0.8.

## Value

An object of class `"geometry_params"`.

## See also

[`term_params()`](https://l-thomson.github.io/thiessen/r/reference/term_params.md)

## Examples

``` r
geometry_params(metric = list("euclidean", "categorical"))
#> geometry_params(metric = list("euclidean", "categorical"), sigma_c = 0.8)
```
