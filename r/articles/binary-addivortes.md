# Binary AddiVortes: probit classification

Binary AddiVortes is the classification variant of Stone and Gosling
(2025): P(y = 1 \| x) = Phi(c + f(x)), fitted with the latent-variable
augmentation of Albert and Chib (1993), as BART’s `pbart` fits its
probit model. It is selected by the
[`probit_outcome()`](https://l-thomson.github.io/thiessen/r/reference/probit_outcome.md)
family. The offset c defaults to Phi^-1(ybar), resolved at fit.

``` r

library(thiessen)

set.seed(1)
n <- 300
x <- cbind(runif(n), runif(n))
label <- factor(ifelse(x[, 1] + 0.3 * x[, 2] + rnorm(n, sd = 0.2) > 0.6,
                       "yes", "no"))

control <- thiessen_control(
  outcome = probit_outcome(),
  tessellations = 25,
  general_params = general_params(burn_in = 100, draws = 200)
)
fit <- thiessen(x, label, control, seed = 1)
#> Warning in thiessen(x, label, control, seed = 1): The chains may not have
#> converged: largest R-hat 1.129 (threshold 1.01), smallest effective sample size
#> 23 (threshold 400). Run more draws or more chains.
fit
#> AddiVortes fit
#> Call: thiessen(x = x, y = label, control = control, seed = 1)
#> probit model, 300 observations, 2 covariates
#> 25 tessellations, 800 draws kept after 100 burn-in, thinning 1
#> In-sample RMSE 0.3619, seed 1
#> 4 chains, largest R-hat 1.129, smallest effective sample size 23
#> Warning: The chains may not have converged: largest R-hat 1.129 (threshold 1.01), smallest effective sample size 23 (threshold 400). Run more draws or more chains.
```

A two-level factor response becomes 0 and 1 with the first level as the
zero, as `glm` treats one; the levels are kept on the fit.
[`predict()`](https://rdrr.io/r/stats/predict.html) returns the
posterior mean probability of the second level, and
`predict(type = "latent")` the latent mean c + f(x) per draw:

``` r

probabilities <- predict(fit, x)
range(probabilities)
#> [1] 0.06117562 0.97740046
mean((probabilities > 0.5) == (label == "yes"))
#> [1] 0.7933333
```

Credible intervals are on the probability scale:

``` r

head(predict(fit, x, interval = "credible", level = 0.9))
#>            fit      lower     upper
#> [1,] 0.2896943 0.13070505 0.5026607
#> [2,] 0.3491636 0.14599493 0.5803786
#> [3,] 0.6775203 0.44608101 0.8809248
#> [4,] 0.9660410 0.89482532 0.9967260
#> [5,] 0.2140054 0.08942802 0.3792176
#> [6,] 0.9444154 0.82475811 0.9957574
```

## What the model fixes

The latent scale is 1 for identification, so
[`sigma()`](https://rdrr.io/r/stats/sigma.html) is empty, a variance
ensemble is not available, and `predict(interval = "prediction")`
errors: a two-point distribution has no continuous predictive interval.
The likelihood in
[`log_lik()`](https://mc-stan.org/rstantools/reference/log_lik.html) is
Bernoulli, and `fit$in_sample_rmse` is the root Brier score.

## References

Albert, J. H. and Chib, S. (1993). Bayesian analysis of binary and
polychotomous response data. *Journal of the American Statistical
Association* 88(422), 669-679.

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. *Journal of Computational and Graphical
Statistics* 34(3), 859-871.
