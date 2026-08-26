# Posterior draws of a fitted model

The variables are `mu[i]`, the mean function at training row i; `sigma`,
where the model has a global residual scale; `cell_count` and
`dimension_count`, the mean cells and mean active covariates per mean
tessellation; and the quantities the experimental models sample where
the fit has them: `df`, the error degrees of freedom under a Student-t
grid; `cutpoint[k]`, the interior cutpoints of the ordinal model;
`bandwidth[j]`, the soft-membership bandwidth of tessellation j;
`inclusion_weight[j]` and `concentration`, the DART inclusion weight of
covariate j and the Dirichlet concentration.

## Usage

``` r
# S3 method for class 'thiessen'
as_draws_df(x, ...)

# S3 method for class 'thiessen'
as_draws_array(x, ...)
```

## Arguments

- x:

  An object of class `"thiessen"`.

- ...:

  Passed to the posterior method.

## Value

A `draws_df`, from
[`posterior::as_draws_df()`](https://mc-stan.org/posterior/reference/draws_df.html).

For `as_draws_array()`, a `draws_array`.

## Details

The chain dimension holds the chains of the fit. A fit of one chain has
one chain, and
[`posterior::summarise_draws()`](https://mc-stan.org/posterior/reference/draws_summary.html)
then reports R-hat as `NA`; effective sample sizes are reported as
usual.

## Examples

``` r
n <- 60
x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
control <- thiessen_control(
  tessellations = 10,
  general_params = general_params(burn_in = 20, draws = 40)
)
fit <- thiessen(x, y, control, seed = 1, chains = 2)
#> Warning: The chains may not have converged: largest R-hat 1.739 (threshold 1.01), smallest effective sample size 4 (threshold 400). Run more draws or more chains.
posterior::summarise_draws(posterior::as_draws_df(fit), "mean", "sd")
#> # A tibble: 63 × 3
#>    variable  mean     sd
#>    <chr>    <dbl>  <dbl>
#>  1 mu[1]    0.401 0.0392
#>  2 mu[2]    0.608 0.0517
#>  3 mu[3]    0.398 0.0341
#>  4 mu[4]    0.602 0.0521
#>  5 mu[5]    0.359 0.0408
#>  6 mu[6]    0.582 0.0349
#>  7 mu[7]    0.342 0.0303
#>  8 mu[8]    0.558 0.0404
#>  9 mu[9]    0.325 0.0339
#> 10 mu[10]   0.517 0.0351
#> # ℹ 53 more rows
```
