# thiessen

Bayesian regression on a sum of Voronoi tessellations (AddiVortes;
Stone and Gosling, 2025, doi:10.1080/10618600.2024.2414104), a variant
of BART (Chipman, George and McCulloch, 2010) in which a cell is a
region of the covariate space rather than a box. The package provides
the Gaussian model of the paper together with its published variants,
Binary AddiVortes (probit classification) and H-AddiVortes
(heteroscedastic variance).

The sampler is the `thiessen` Rust crate, built from sources vendored
in the package. The method and all credit for it belong to its
authors; this package is an independent implementation, and its test
suite compares posterior summaries against the authors' R package,
[AddiVortes](https://github.com/johnpaulgosling/AddiVortes).

## Installation

The package is not yet on CRAN. Building from source needs a Rust
toolchain (rustc 1.74 or later with Cargo):

``` r
# install.packages("remotes")
remotes::install_github("l-thomson/thiessen", subdir = "r")
```

## Example

``` r
library(thiessen)

n <- 200
x <- cbind(runif(n), runif(n))
y <- 2 * (x[, 1] - 0.5)^2 + 0.5 * x[, 2] + rnorm(n, sd = 0.1)

fit <- thiessen(x, y, seed = 1)
predict(fit, x)
plot(fit)
```

A formula interface takes data frames, with factor covariates encoded
as `model.matrix` encodes them, and `chains = 2` or more adds R-hat
and effective sample size diagnostics. `posterior::as_draws_df()`,
`posterior_predict()`, `log_lik()` and the other established generics
work on a fit; see `vignette("thiessen")`.

## Priors

The priors are those of Stone and Gosling (2025) and are set through
`thiessen_control()`. The outcome family carries the prior on the
noise: `gaussian(nu, q)` scales the inverse chi-squared prior on
sigma^2 so that a proportion `q` of its mass lies below the sample
variance of the response. The mean ensemble's priors sit in
`term_params()`: `k` fixes the spread of the cell-value prior,
`lambda_c` is the Poisson rate of the cell-count prior, and the nested
`geometry_params()` and `structure_params()` hold the centre-proposal
scale `sigma_c` and the covariate-inclusion weight `omega`. Setting
`prior_only = TRUE` in `general_params()` draws from the prior alone.
`vignette("control-surface")` walks the whole surface;
`vignette("sampler-api")` drives the sampler one sweep at a time.

## References

Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
Bayesian additive regression trees. The Annals of Applied Statistics
4(1), 266-298. doi:10.1214/09-AOAS285

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. Journal of Computational and Graphical
Statistics 34(3), 859-871. doi:10.1080/10618600.2024.2414104
