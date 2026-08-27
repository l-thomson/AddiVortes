# Per-draw sampler diagnostics of a fitted model

The trace of the quantities the sampler records once per kept draw. For
a draws object the same quantities, with the mean function, come from
[`posterior::as_draws_df()`](https://mc-stan.org/posterior/reference/draws_df.html),
which
[`bayesplot::mcmc_trace()`](https://mc-stan.org/bayesplot/reference/MCMC-traces.html)
takes.

## Usage

``` r
thiessen_diagnostics(object)
```

## Arguments

- object:

  An object of class `"thiessen"`.

## Value

A data frame with one row per kept draw and the columns `chain`, the
chain the draw comes from; `draw`, its index within that chain; `sigma`,
the residual standard deviation, under the Gaussian model only;
`cell_count`, the mean cells per mean tessellation; and
`dimension_count`, the mean active covariates per mean tessellation.

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
#> Warning: The chains may not have converged: largest R-hat 1.777 (threshold 1.01), smallest effective sample size 7 (threshold 400). Run more draws or more chains.
head(thiessen_diagnostics(fit))
#>   chain draw      sigma cell_count dimension_count
#> 1     1    1 0.07601317        2.8               2
#> 2     1    2 0.07994068        2.8               2
#> 3     1    3 0.07728484        2.9               2
#> 4     1    4 0.06098559        2.9               2
#> 5     1    5 0.07432584        2.8               2
#> 6     1    6 0.06930679        2.9               2
```
