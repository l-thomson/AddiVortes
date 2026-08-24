# Trace plots of a fitted model

The per-draw quantities of
[`thiessen_diagnostics()`](https://l-thomson.github.io/thiessen/r/reference/thiessen_diagnostics.md)
as traces, one panel per quantity and one line per chain: `sigma` where
the model has one, the mean cells per mean tessellation and the mean
active covariates per mean tessellation. Burn-in sweeps are discarded
before the first draw is kept, so a trace shows the kept draws only.

## Usage

``` r
# S3 method for class 'thiessen'
plot(x, ...)
```

## Arguments

- x:

  An object of class `"thiessen"`.

- ...:

  Ignored.

## Value

`x`, invisibly.

## Details

For traces of the mean function and for distributional displays, pass
[`posterior::as_draws_df()`](https://mc-stan.org/posterior/reference/draws_df.html)
to bayesplot: `mcmc_trace()` plots sequences, `mcmc_areas()` and
`mcmc_dens()` plot posterior densities, and `mcmc_combo()` plots both in
one figure.

## Examples

``` r
n <- 60
x <- cbind(seq(0, 1, length.out = n), rep(c(0, 0.5), length.out = n))
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2]
control <- thiessen_control(
  tessellations = 10,
  general_params = general_params(burn_in = 20, draws = 40)
)
plot(thiessen(x, y, control, seed = 1))
```
