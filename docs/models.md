# Models

The observation models the crate fits, selected by `Config::outcome`
and, for H-AddiVortes, `variance_params.num_tessellations`. Each
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

## Gaussian (`Outcome::Gaussian`)

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

## Probit (`Outcome::Probit`)

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

## Heteroscedastic (`Outcome::Gaussian` with `variance_params.num_tessellations` above 0)

H-AddiVortes; the structure is that of HBART (Pratola, Chipman, George
and McCulloch 2020):

    y_i = f(x_i) + s(x_i) e_i,   e_i ~ N(0, 1),
    f(x) = sum_{j=1}^m g(x; T_j, M_j),   s^2(x) = prod_{j=1}^{m'} v(x; T'_j, V_j),

v(x; T', V) the cell value of the cell of variance tessellation T' that x
falls in. The mean ensemble is that of the Gaussian model with
per-observation precision 1 / s^2(x_i); the variance ensemble has
inverse-gamma cell values and a multiplicative backfit. One sweep updates
the variance ensemble given the residuals y - f, then the mean ensemble
(the order of the authors' code; HBART sweeps mean then variance, both
valid Gibbs orders).

Priors: mean cells mu ~ N(0, sigma_mu^2), sigma_mu = 0.5 / (k sqrt m) on
the response scaled to [-0.5, 0.5]. Each variance cell
v ~ Inv-Gamma(nu' / 2, nu' lambda' / 2) with

    lambda' = lambda^(1 / m'),   nu' = 2 / (1 - (1 - 2 / nu)^(1 / m')),

lambda calibrated as for the Gaussian model. These make the prior mean of
s^2(x), (nu' lambda' / (nu' - 2))^m', equal to nu lambda / (nu - 2), the
prior mean of the Gaussian model's sigma^2; without the matching the
product of m' cell values has a mean and spread that grow with m' (HBART
s. 3.2). nu > 2 is required. The variance ensemble shares lambda_c, omega
and sigma_c with the mean ensemble; every variance cell starts at
sigma_hat^(2 / m'), so s^2 starts at sigma_hat^2. Configuration:
`Config::new().with_m_var(40)`; the paper's m' is 40.

Fitted model: `predict` is the posterior mean of f(x); `predict_variance`
is s_d^2(x) per draw on the caller's scale (the square of `rbart`'s
`sdraws`); `prediction_interval` and `log_likelihood` use
N(f_d(x), s_d^2(x)); `sigma` is empty.

Correspondence with `rbart` (HBART): m = `ntree`, m' = `ntreeh`, k = `k`,
nu = `overallnu`, sigma_hat = `overallsd`, burn_in = `nskip`, draws =
`ndpost`; `rbart` places the (nu', lambda') matching on the same
quantities. With the authors' script: m = `m`, m' = `m_var`, nu = `nu`,
q = `q`, k = `k`, sigma_c = `sd`, omega = `Omega`, lambda_c =
`lambda_rate` (the script defaults to 25; the crate's default is 5 for
every model), burn_in = `burn_in`, draws = `max_iter - burn_in`. The
script is pre-0.6.8 and the comparison against it is informational
(`benchmarks/upstream/heteroscedastic_variant.R`).

## Distances

`Config::metric` names the metric of each covariate column (CRAN
AddiVortes `metric`, with `members` grouping the columns of a sphere);
empty means Euclidean throughout. An observation is assigned to the
nearest centre over a tessellation's active columns, the distance being
the sum over metrics of their squared distances; ties go to the lowest
centre index.

| `Metric`                     | CRAN AddiVortes           | distance                         | column scale                | centre coordinate law                                         |
|------------------------------|---------------------------|----------------------------------|-----------------------------|---------------------------------------------------------------|
| `Euclidean`                  | `"E"` (default)           | squared Euclidean                | min-max to [-0.5, 0.5]      | N(0, sigma_c^2)                                               |
| `Spherical { sphere }`       | `"S"`, `members = sphere` | squared great-circle angle       | radians, unscaled           | N(mid, sd^2), sd = range / (2 Phi^-1(0.75)); longitude wrapped to [-pi, pi] |
| `Categorical`                | `"C"`, `cat.onehot = FALSE` | 2 / n^2 per mismatching column | integer level codes, unscaled | uniform over the n training levels                          |

Spherical: the columns declared for a sphere are its latitudes and,
last, its longitude, the coordinate with period 2 pi (upstream moves the
column of range above pi last; here the declaration order is the
contract, and a latitude spanning more than pi is rejected at fit). The
angle between a row and a centre follows the spherical law of cosines,
cos c = sin a sin b + cos a cos b cos(Delta longitude), nested over the
latitudes for spheres of more than two columns; a sphere of one column
is a circle and takes the shorter arc. When a tessellation uses only
some of a sphere's columns, the centre takes the row's own coordinates
in the others, so the distance is measured along the row's parallel or
meridian (upstream `knnx_index_cpp`). The per-column mean and standard
deviation of the coordinate law are upstream's `mus` and `sd` for
non-Euclidean columns; the longitude is wrapped by upstream's
`period_shift`. Several spheres take distinct labels; their angles add.

Columns of a sphere are separate covariates for the dimension prior and
the structural moves, as upstream.

Categorical: the levels of a column are its distinct training values, so
0-based and 1-based codes both serve (upstream uses `as.numeric(factor)`,
1..n, and takes n as the largest code); a code unseen in training is a
mismatch against every centre. The weight 2 / n^2 is the mismatch weight
of Eskin et al. (2002) as upstream states it (Eskin's distance proper is
2 / (n^2 + 2)). CRAN AddiVortes 0.6.9 evaluates `2 / (ncat * ncat)` in
integer arithmetic, which is 0 for n >= 2, so its `metric = "C"` assigns
every row to the first centre on categorical columns; no comparison
fixture exists until that is corrected. Upstream's default path,
`cat.onehot = TRUE`, encodes a factor as d - 1 indicators taking values
{0, `catScaling`}; the bindings' indicator columns are Euclidean and
min-max scaled, which is `catScaling = 1` without upstream's clamp of
centre proposals to [0, `catScaling`].

## Experimental models

Models behind the `experimental` Cargo feature are stated here under
their own headings as they land; their status is kept only in the table
in [experimental.md](experimental.md).

## References

- Albert, J. H. and Chib, S. (1993). Bayesian analysis of binary and
  polychotomous response data. Journal of the American Statistical
  Association 88(422), 669-679.
- Eskin, E., Arnold, A., Prerau, M., Portnoy, L. and Stolfo, S. (2002).
  A geometric framework for unsupervised anomaly detection. In
  Applications of Data Mining in Computer Security, 77-101. Springer.
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
