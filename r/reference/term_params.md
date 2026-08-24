# One ensemble of tessellations

The size, priors and covariate space of one ensemble.
[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md)
takes one as `mean_params` and, for the heteroscedastic model, one as
`variance_params`.

## Usage

``` r
term_params(
  tessellations = NULL,
  k = 3,
  lambda_c = 5,
  geometry = NULL,
  structure = NULL
)
```

## Arguments

- tessellations:

  Number of tessellations in the ensemble. `NULL`, the default, resolves
  at fit to 200 as `mean_params` and to 0 as `variance_params`; a
  positive count as `variance_params` selects the heteroscedastic model
  (the paper's count is 40).

- k:

  Cell-value prior spread k: sigma_mu = w / (k sqrt(m)) with the
  half-width w the outcome family owns (Chipman, George and McCulloch
  2010, s. 4). Default 3. The variance ensemble's inverse-gamma cells do
  not use it.

- lambda_c:

  Cell-count prior rate lambda_c: b - 1 ~ Poisson(lambda_c). Default 5,
  following AddiVortes 0.6.8 and later; the paper reports 25.

- geometry:

  The covariate space, from
  [`geometry_params()`](https://l-thomson.github.io/thiessen/r/reference/geometry_params.md).
  `NULL`, the default, takes the core's defaults. The ensembles share
  one covariate space: set it on `mean_params` and it applies to
  `variance_params` as well.

- structure:

  The covariate-inclusion prior, from
  [`structure_params()`](https://l-thomson.github.io/thiessen/r/reference/structure_params.md).
  `NULL` takes the core's defaults. Shared between the ensembles like
  `geometry`.

## Value

An object of class `"term_params"`.

## See also

[`thiessen_control()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_control.md),
[`geometry_params()`](https://l-thomson.github.io/thiessen/r/reference/geometry_params.md),
[`structure_params()`](https://l-thomson.github.io/thiessen/r/reference/structure_params.md)

## Examples

``` r
term_params(tessellations = 200, lambda_c = 25)
#> term_params(tessellations = 200, k = 3, lambda_c = 25)
```
