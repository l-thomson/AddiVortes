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
fit
#> AddiVortes fit
#> Call: thiessen(x = x, y = label, control = control, seed = 1)
#> probit model, 300 observations, 2 covariates
#> 25 tessellations, 200 draws kept after 100 burn-in, thinning 1
#> In-sample RMSE 0.362, seed 1
#> 1 chain; R-hat and effective sample sizes need two or more chains
```

A two-level factor response becomes 0 and 1 with the first level as the
zero, as `glm` treats one; the levels are kept on the fit.
[`predict()`](https://rdrr.io/r/stats/predict.html) returns the
posterior mean probability of the second level, and
`predict(type = "latent")` the latent mean c + f(x) per draw:

``` r

probabilities <- predict(fit, x)
range(probabilities)
#> [1] 0.05785467 0.97805843
mean((probabilities > 0.5) == (label == "yes"))
#> [1] 0.7866667
```

Credible intervals are on the probability scale:

``` r

head(predict(fit, x, interval = "credible", level = 0.9))
#>            fit      lower     upper
#> [1,] 0.2493825 0.11678199 0.4562457
#> [2,] 0.3883306 0.15240819 0.6389193
#> [3,] 0.7034044 0.49475748 0.8891872
#> [4,] 0.9594971 0.89170780 0.9950113
#> [5,] 0.2085537 0.09022212 0.3620440
#> [6,] 0.9408933 0.81420680 0.9948472
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
