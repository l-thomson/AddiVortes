# H-AddiVortes: heteroscedastic regression

H-AddiVortes is the heteroscedastic variant of Stone and Gosling (2025):
y = f(x) + e with e ~ N(0, s^2(x)), where the variance s^2(x) is a
multiplicative ensemble of inverse-gamma variance tessellations, the
structure of HBART (Pratola, Chipman, George and McCulloch, 2020). It is
selected by attaching a variance ensemble to the Gaussian family: a
positive tessellation count on `variance_params`. The paper’s count is
40.

``` r

library(thiessen)

set.seed(1)
n <- 300
x <- cbind(runif(n), runif(n))
# The noise scale grows with the first covariate.
y <- x[, 1] + rnorm(n, sd = 0.05 + 0.3 * x[, 1])

control <- thiessen_control(
  mean_params = term_params(tessellations = 25),
  variance_params = term_params(tessellations = 10),
  general_params = general_params(burn_in = 100, draws = 200)
)
fit <- thiessen(x, y, control, seed = 1)
#> Warning in thiessen(x, y, control, seed = 1): The chains may not have
#> converged: largest R-hat 1.202 (threshold 1.01), smallest effective sample size
#> 15 (threshold 400). Run more draws or more chains.
fit
#> AddiVortes fit
#> Call: thiessen(x = x, y = y, control = control, seed = 1)
#> heteroscedastic model, 300 observations, 2 covariates
#> 25 tessellations, 800 draws kept after 100 burn-in, thinning 1
#> In-sample RMSE 0.2027, seed 1
#> 4 chains, largest R-hat 1.202, smallest effective sample size 15
#> Warning: The chains may not have converged: largest R-hat 1.202 (threshold 1.01), smallest effective sample size 15 (threshold 400). Run more draws or more chains.
```

`predict(type = "variance")` returns s^2(x) per draw; its posterior mean
recovers the rising noise scale:

``` r

variance <- colMeans(predict(fit, x, type = "variance"))
order <- order(x[, 1])
c(
  low_x = mean(variance[head(order, 50)]),
  high_x = mean(variance[tail(order, 50)])
)
#>       low_x      high_x 
#> 0.009572963 0.093159358
```

Predictive intervals widen where the variance is high, which is the
point of the model:

``` r

interval <- predict(fit, x, interval = "prediction", level = 0.9)
width <- interval[, "upper"] - interval[, "lower"]
widths <- c(
  low_x = mean(width[head(order, 50)]),
  high_x = mean(width[tail(order, 50)])
)
# The claim above, asserted rather than read: a wrong number fails the build.
stopifnot(widths[["high_x"]] > widths[["low_x"]])
widths
#>     low_x    high_x 
#> 0.3571411 1.0476867
```

## What is shared and what is separate

The two ensembles declare one covariate space: `geometry` and
`structure` set on `mean_params` apply to `variance_params` as well. The
variance ensemble needs a sampled sigma^2 to carry, so it requires the
Gaussian family with nu \> 2, and it is not available under the probit
family, whose latent scale is fixed at 1 for identification.
[`sigma()`](https://rdrr.io/r/stats/sigma.html) is empty under this
model; the spread is `predict(type = "variance")`.

## References

Pratola, M. T., Chipman, H. A., George, E. I. and McCulloch, R. E.
(2020). Heteroscedastic BART via multiplicative regression trees.
*Journal of Computational and Graphical Statistics* 29(2), 405-417.

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. *Journal of Computational and Graphical
Statistics* 34(3), 859-871.
