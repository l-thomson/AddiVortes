# Variable inclusion proportions of a fitted model

The share of the active dimensions of the mean tessellations falling on
each covariate, averaged over the kept draws, as `dbarts::varcount`
summarises tree splits. The values sum to one.

## Usage

``` r
variable_inclusion(object)
```

## Arguments

- object:

  An object of class `"thiessen"`.

## Value

A numeric vector, one value per column of the design, named as the
design columns are and unnamed where the design has no column names.

## Details

They report where the ensemble spent its dimensions, not which
covariates carry signal, and they inherit the covariate-inclusion prior:
at the default `omega` of `min(3, p)` every dimension is always active
when p is 3 or fewer, so the proportions are then uniform by
construction, exactly 1/p. Separation is weak at p = 4, where two
informative covariates measured 0.26 and 0.29 against pure noise at 0.25
and 0.19. Do not read them as variable selection.

## References

Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
Bayesian additive regression trees. *The Annals of Applied Statistics*
4(1), 266-298.
[doi:10.1214/09-AOAS285](https://doi.org/10.1214/09-AOAS285) , s. 5.2.

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
variable_inclusion(fit)
#> [1] 0.5 0.5
```
