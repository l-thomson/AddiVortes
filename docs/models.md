# Models

The observation models the crate fits, selected by `Config::model`. Each
entry states the model, its priors, what is fixed rather than estimated,
the correspondence of its parameters with the paper and with the
BART-family reference implementation, and what the fitted model returns.
The rustdoc pages under `thiessen::models` carry the same statements.

Shared across models: the mean function f(x) = sum_{j=1}^m g(x; T_j, M_j)
is a sum of m Voronoi tessellations, each over a random subset of the
covariates, with cell means mu ~ N(0, sigma_mu^2); cells b - 1 ~
Poisson(lambda_c); active covariates d - 1 ~ Binomial(p - 1, omega / p);
centre coordinates N(0, sigma_c^2) in the covariate space scaled to
[-0.5, 0.5] per column (Stone and Gosling 2025, s. 2.3). The sampler is
the backfitting Gibbs sampler of Algorithm 1 with the six structural
Metropolis-Hastings moves of Appendix B, carrying the corrections CRAN
AddiVortes made in 0.6.8.

## Gaussian (`Model::Gaussian`)

    y_i = f(x_i) + e_i,   e_i ~ N(0, sigma^2)

Priors: sigma_mu = 0.5 / (k sqrt m) on the response scaled to
[-0.5, 0.5]; sigma^2 ~ nu lambda / chi^2_nu with lambda set so that
Pr(sigma < sigma_hat) = q, sigma_hat the residual standard deviation of a
least-squares fit (the sample standard deviation when that fit is
degenerate). Nothing is fixed.

| crate       | paper (s. 2.3) | CRAN AddiVortes            | BART `wbart` |
|-------------|----------------|----------------------------|--------------|
| `m`         | m              | `m`                        | `ntree`      |
| `nu`        | nu             | `nu`                       | `sigdf`      |
| `q`         | q              | `q`                        | `sigquant`   |
| `k`         | k              | `k`                        | `k`          |
| `sigma_c`   | sigma_c        | `sd`                       | none         |
| `omega`     | omega          | `Omega`                    | none         |
| `lambda_c`  | lambda_c       | `LambdaRate`               | none         |
| `burn_in`   |                | `mcmcBurnIn`               | `nskip`      |
| `draws`     |                | `totalMCMCIter - mcmcBurnIn` | `ndpost`   |
| `thinning`  |                |                            | `keepevery`  |

Defaults: m = 200, nu = 6, q = 0.85, k = 3, sigma_c = 0.8, omega =
min(3, p), lambda_c = 5 (CRAN AddiVortes >= 0.6.8; the paper reports 25,
available as `Config::paper()`).

Fitted model: `predict` is the posterior mean of f(x); `predict_variance`
is sigma_d^2 per draw, constant across rows; `prediction_interval` and
`log_likelihood` use N(f_d(x), sigma_d^2); `sigma` is sigma_d on the
caller's scale.

## Probit (`Model::Probit`)

    y_i in {0, 1},   P(y_i = 1 | x_i) = Phi(c + f(x_i))

Sampler: the data augmentation of Albert and Chib (1993). A latent
z_i ~ N(c + f(x_i), 1) truncated to the sign of y_i is refreshed before
each sweep (truncated normal by the Robert 1995 exponential rejection),
and the mean ensemble is updated as in the Gaussian model with z - c as
the response and unit variance (Chipman, George and McCulloch 2010,
s. 4).

Priors: sigma_mu = 3 / (k sqrt m) on the latent scale, so that the prior
on f(x) puts high probability on [-3, 3] (Chipman, George and McCulloch
2010, s. 4; BART `pbart`). The response is not scaled; the covariates are
scaled as in the Gaussian model.

Fixed: the latent variance is 1 (not identified in a probit model; no
sigma^2 is drawn or reported). The offset c is `Config::offset`, default
Phi^-1(ybar) (BART `binaryOffset`); the initial state is f = 0, the offset
carrying the mean. Chipman, George and McCulloch (2010) centre at c = 0.

| crate      | Binary AddiVortes script | BART `pbart`   |
|------------|--------------------------|----------------|
| `m`        | `m`                      | `ntree`        |
| `k`        | `k`                      | `k`            |
| `offset`   | none (c = 0)             | `binaryOffset` |
| `sigma_c`  | `var`                    | none           |
| `omega`    | `Omega`                  | none           |
| `lambda_c` | `lambda_rate`            | none           |
| `burn_in`  | `burn_in`                | `nskip`        |
| `draws`    | `max_iter - burn_in`     | `ndpost`       |
| `thinning` |                          | `keepevery`    |

Defaults stay the crate's (k = 3, m = 200). `pbart` defaults to k = 2 and
`ntree = 50`; a reader porting settings from BART should set both. The
authors' script (Adam-Stone2/Binary_AddiVortes at `ffc914b`) uses
sigma_mu = 3 / (k sqrt m) as here, has no offset, initialises the latent
fit at ybar (a probability on the wrong scale), and carries the
structural-move terms CRAN corrected in 0.6.8; its comparison is
informational (`benchmarks/upstream/binary_variant.R`).

Fitted model: `predict` is the posterior mean of P(y = 1 | x) and
`predict_draws` its per-draw values; `predict_latent` is c + f(x) per
draw; `credible_interval` is on the probability scale; `log_likelihood`
is the Bernoulli log-likelihood; `prediction_interval` and
`predict_variance` return `Error::NotApplicable`; `sigma` is empty;
`in_sample_rmse` is the root Brier score. Input: labels in {0, 1} with
both present; anything else is `Error::InvalidLabel`.

## Heteroscedastic (`Model::Heteroscedastic`)

In development (OSS-69). The engine carries the variance ensemble: s^2(x)
is the product of `m_var` variance tessellations with inverse-gamma cell
values, the structure of HBART (Pratola, Chipman, George and McCulloch
2020).

## References

- Albert, J. H. and Chib, S. (1993). Bayesian analysis of binary and
  polychotomous response data. Journal of the American Statistical
  Association 88(422), 669-679.
- Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
  Bayesian additive regression trees. Annals of Applied Statistics 4(1),
  266-298.
- Pratola, M. T., Chipman, H. A., George, E. I. and McCulloch, R. E.
  (2020). Heteroscedastic BART via multiplicative regression trees.
  Journal of Computational and Graphical Statistics 29(2), 405-417.
- Robert, C. P. (1995). Simulation of truncated normal variables.
  Statistics and Computing 5, 121-125.
- Sparapani, R., Spanbauer, C. and McCulloch, R. (2021). Nonparametric
  machine learning and efficient computation with Bayesian additive
  regression trees: the BART R package. Journal of Statistical Software
  97(1), 1-66.
- Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
  Voronoi tessellations. Journal of Computational and Graphical
  Statistics 34(3), 859-871.
