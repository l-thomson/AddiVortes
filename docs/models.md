# Models

The observation models the crate fits, selected by `Config::outcome`
and, for H-AddiVortes, `variance_params.tessellations`. Each
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

## Heteroscedastic (`Outcome::Gaussian` with `variance_params.tessellations` above 0)

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

## The supported-likelihood boundary

The crate supports observation models that admit an exact conditionally
Gaussian augmentation: likelihoods under which, given latent variables
with known conditional laws, the response is Gaussian with known
variance, so the backfitting sweep and the marginal likelihood of every
structural move stay in closed form. The probit model is the canonical
case (Albert and Chib 1993). A Student-t likelihood is in scope as an
exact scale mixture of normals (Geweke 1993); a Poisson or negative
binomial model is reachable only through an approximating mixture
(Fruhwirth-Schnatter et al. 2009) and would be approximate by
construction, which its statement would have to say. Gamma, Beta,
Weibull and Tweedie responses have no such augmentation and are out of
scope permanently: carrying one would mean a second sampling kernel, not
a missing file. A need that is not an outcome model or a setting goes
through the sampler API's response seam and stays there until it earns a
name.

## Augmentation groups

Sequencing prose, not API structure: the candidate models group by the
machinery their augmentation needs. Latent-response models (probit, and
the logit family through Polya-Gamma mixing, Polson, Scott and Windle
2013) share the latent refresh against a fixed ensemble; scale-mixture
models (Student-t, Laplace, contaminated normal) share per-observation
variance draws. The first model of each group carries the shared
machinery and the rest reuse it. The contaminated normal is the worked
example of why the groups stay prose: it is a scale mixture, and its
mixture indicator is also a latent label, so a group field on the
configuration would have to pick one parent for a model that has two.
The configuration names models; it never names groups.

## Validation

The configuration surface is a parameter space, and simulation-based
calibration cannot cover it exhaustively, so the claim is stated
exactly. The configurations listed in [calibrated.md](calibrated.md) are
covered by the calibration suite: simulation-based calibration and
Geweke tests at two sizes, with broken-sampler fixtures showing the
gates reject a miscalibrated kernel (`docs/testing.md`). Component
options are additionally verified in isolation by bit-exact equivalence,
known-answer and property tests. Every other combination of the
documented options is valid to run and is not separately verified.

## Experimental models

Models behind the `experimental` Cargo feature are stated here under
their own headings as they land; their status is kept only in the table
in [experimental.md](experimental.md).

### DART inclusion (`structure.inclusion` entry `dart`, experimental)

Not a model: in the API it is a value of the term group's inclusion
prior, following the BART package, which ships it as `sparse = TRUE`
with `a`, `b` and `rho` rather than as a separate function. It is
stated here because it is model-grade in validation: the inclusion
weights are sampled, which changes the posterior.

    s ~ Dirichlet(theta / p, ..., theta / p),
    P(S | d, s) = prod_{j in S} s_j / e_d(s),
    lambda = theta / (theta + rho),  lambda uniform on a 1000-point grid
    of (0, 1) with prior weights Beta(a, b).

e_d is the elementary symmetric polynomial, the subset-prior
normaliser. Defaults a = 0.5, b = 1, rho = p (Linero 2018; the BART
package's `sparse = TRUE` defaults). The grid is the prior, not an
approximation of one: theta's conditional is sampled exactly on it. s
is updated by a Metropolis step whose Dirichlet(theta / p + counts)
proposal leaves exactly the normalisers e_d in the acceptance ratio.
The weights are shared between the mean and the variance ensembles,
whose declared structure is one group. Validation: SBC and Geweke at
both sizes, and a broken-sampler fixture dropping the normalisers from
the weight update.

### Linear cell basis (`cell.basis` entry `linear`, experimental)

Not a model: a value of the mean term group's within-cell response
surface. It is stated here because it is model-grade in validation: the
cell conjugate update changes, so it changes the posterior.

    g(x; T, M) = mu_k + beta_k' (x_A - c_k)   for x in cell k,
    (mu_k, beta_k) ~ N(0, sigma_mu^2 I_(d+1)),

with x_A the active coordinates and c_k the cell's centre, so mu keeps
its role as the level at the centre. The cell update draws (mu, beta)
jointly from the (d + 1)-dimensional conjugate normal, and the
structural moves integrate the whole coefficient vector out: per cell
-ln det(I + sigma_mu^2 A_k) / 2 + b_k' (Sigma_0^-1 + A_k)^-1 b_k / 2,
the (d + 1)-dimensional form of the constant-basis expression, with
A_k, b_k the weighted normal equations of the within-cell design
u = (1, x_A - c_k). Precedent for linear leaves in a BART-family
ensemble: Prado, Moral and Parnell (2021). Needs min-max scaled columns
(checked at fit); mean slot only, the variance ensemble's inverse-gamma
cells keep the constant basis. Validation: known-answer tests of the
marginal and the joint draw, SBC and Geweke at both sizes, and a
broken-sampler fixture dropping the determinant term.

### Soft membership (`geometry.membership` entry `soft`, experimental)

Not a model: a value of the mean term group's geometry, softening the
nearest-centre assignment the way SBART softens the tree split (Linero
and Yang 2018; the SoftBart package). It is stated here because it is
model-grade in validation: each observation loads on every cell, so the
cell conjugate update changes, which changes the posterior.

    w_k(x) proportional to exp(-d_k^2(x) / (2 tau^2)),  sum_k w_k(x) = 1,
    g(x; T, M) = sum_k w_k(x) mu_k,
    tau ~ Exponential(rate),  rate 10 by default,

with d_k^2 the squared distance of the active metrics to centre k (the
hard rule's own key) and tau a per-tessellation bandwidth on the scaled
covariate space, so the prior mean bandwidth is a tenth of a column's
range (the SoftBart `tau_rate` default). As tau falls to 0 the weights
recover the hard assignment; the difference from SBART is that the
kernel acts on the distance to each centre rather than on a chain of
split gates. The cell update draws the mu vector jointly from the
b-dimensional conjugate normal with W'DW + I / sigma_mu^2 the
precision, and the structural moves integrate it out:
-ln det(I + sigma_mu^2 W'DW) / 2 + b' (Sigma_0^-1 + W'DW)^-1 b / 2, one
block for the whole tessellation, W the n x b weight matrix, D the
observation precisions and b = W'Dr. tau is updated by a random-walk
Metropolis step on ln tau against that integrated likelihood. The
empty-cell rule still counts nearest-centre members. Constant cell
basis and constant spread only: the linear basis has no derived
weighted update, and the variance ensemble's inverse-gamma cells have
no closed-form weighted conditional; the probit model composes.
Validation: known-answer tests of the marginal and the joint draw, SBC
and Geweke at both sizes, and a broken-sampler fixture dropping the
determinant term.

### Tobit (`Outcome::Tobit`, experimental)

    y*_i = f(x_i) + e_i,   e_i ~ N(0, sigma^2),
    y_i  = lower  if y*_i <= lower,
           upper  if y*_i >= upper,
           y*_i   otherwise

The type-I tobit model (Tobin 1958): the limits are known constants of
the design, a response value equal to a limit is read as censored on
that side, and a value beyond a limit is rejected at fit
(`Error::ResponseBeyondLimit`). At least one limit is declared; models
with unknown censoring points are out of scope.

Sampler: the data augmentation of Chib (1992). Each censored row's
latent is refreshed before the sweep from N(f(x_i), sigma^2) truncated
to its censored side (Robert 1995 exponential rejection), an observed
row's latent being its response, and the completed response runs the
Gaussian model's sweep unchanged. The scan is y* | sigma^2, f then
sigma^2 | y* then f | y*, sigma^2; the refresh runs first so sigma^2
and the ensemble only condition on latents drawn from their
conditional. No structural move gains an acceptance-ratio term. With a
variance ensemble attached the truncated draw's variance is s^2(x_i),
the same per-observation precision the backfit uses.

Priors: the Gaussian model's exactly, on the response min-max scaled
to [-0.5, 0.5]; the limits cross by the same frozen affine map;
sigma_hat calibrates from the observed response with censored rows at
their limits. Imputed latents may fall outside the training range,
which the frozen map permits. Configuration:
`Outcome::tobit(lower, upper)`, each limit optional, nu and q on the
tobit parameters; uncensored data (no row at a limit) reproduces the
Gaussian model draw for draw at the same seed.

Correspondence: with MCMCpack `MCMCtobit` (the Chib 1992 sampler for
the linear model), lower = `below`, upper = `above` (its defaults are 0
and infinity).

Fitted model: `predict` is the posterior mean of the uncensored f(x);
`predict_variance` is sigma_d^2 (s_d^2(x) under a variance ensemble);
`prediction_interval` is the censored predictive's central interval,
the uncensored ends clamped to the limits; `log_likelihood` is the
type-I tobit likelihood (ln Phi((lower - f_d) / s_d) at a row censored
below, ln Phi((f_d - upper) / s_d) above, the Normal log density
otherwise); `sigma` is sigma_d on the caller's scale; `in_sample_rmse`
is against the observed response, censored rows at their limits.
Validation: fixed-tessellation quadrature known-answer test against
numerical integration of the censored likelihood, SBC and Geweke at
both sizes.

### AFT survival (`Outcome::Aft`, experimental)

    ln T_i = f(x_i) + e_i,   e_i ~ N(0, sigma^2),
    observed (t_i, delta_i): delta_i = 1 is an event (T_i = t_i),
                             delta_i = 0 right-censoring (T_i > t_i)

The lognormal accelerated failure time model (Wei 1992), the model of
the BART package's `abart`. The times are positive
(`Error::InvalidSurvivalTime`), the event indicator is one flag per row
(`Error::EventCountMismatch`), and both are data: the model is fitted
through `fit_aft` or `Sampler::aft`, and `fit` rejects the outcome.
Right-censoring only; interval censoring is a separate model.

Sampler: censored-data augmentation on the log scale, the censored
refresh shared with the tobit model. Each censored row's latent log
time is refreshed before the sweep from N(f(x_i), sigma^2) truncated to
[ln t_i, inf) (Robert 1995 exponential rejection), an event row's
latent being ln t_i, and the completed log-time response runs the
Gaussian model's sweep unchanged, with the tobit model's scan order
(refresh first). No structural move gains an acceptance-ratio term.
With a variance ensemble the truncated draw's variance is s^2(x_i).

Priors: the Gaussian model's exactly, on ln t min-max scaled to
[-0.5, 0.5]; a censored row's truncation point is its own scaled
ln t_i, so nothing beyond the response crosses the frozen map;
sigma_hat calibrates from the observed log times, censored rows at
their censoring values. All-event data reproduces the Gaussian model
on ln t draw for draw at the same seed.

| crate      | BART `abart`  |
|------------|---------------|
| `times`    | `times`       |
| `events`   | `delta`       |
| `m`        | `ntree`       |
| `k`        | `k`           |
| `nu`       | `sigdf`       |
| `q`        | `sigquant`    |
| `burn_in`  | `nskip`       |
| `draws`    | `ndpost`      |
| `thinning` | `keepevery`   |

`abart` defaults to k = 2, `sigdf = 3` and `sigquant = 0.90`; the crate
keeps its own defaults (k = 3, nu = 6, q = 0.85). `abart` centres the
response with an offset; here the min-max response map carries the
centring. The comparison against `abart` on a fixed dataset is
informational (`benchmarks/upstream/aft_abart.R`): trees and
tessellations are different priors, so the posteriors are close but
not equal.

Fitted model: `predict` is the posterior mean of f(x) on the log-time
scale (`abart`'s `yhat`); `predict_variance` is sigma_d^2 on the log
scale (s_d^2(x) under a variance ensemble); `prediction_interval` is
the predictive interval of a new log time; `log_likelihood` is
`Error::NotApplicable` (the pointwise likelihood needs the event
indicator) and `log_likelihood_survival(x, times, events)` takes it:
ln N(ln t_i; f_d, s_d^2) at an event, ln Phi((f_d - ln t_i) / s_d) at
a censored row; `sigma` is sigma_d times the training range of ln t;
`in_sample_rmse` is on the log-time scale against the observed log
times, censored rows at their censoring values. Validation:
fixed-tessellation quadrature known-answer test against numerical
integration of the model's own censored likelihood, SBC and Geweke at
both sizes.

## References

- Albert, J. H. and Chib, S. (1993). Bayesian analysis of binary and
  polychotomous response data. Journal of the American Statistical
  Association 88(422), 669-679.
- Eskin, E., Arnold, A., Prerau, M., Portnoy, L. and Stolfo, S. (2002).
  A geometric framework for unsupervised anomaly detection. In
  Applications of Data Mining in Computer Security, 77-101. Springer.
- Fruhwirth-Schnatter, S., Fruhwirth, R., Held, L. and Rue, H. (2009).
  Improved auxiliary mixture sampling for hierarchical models of
  non-Gaussian data. Statistics and Computing 19, 479-492.
- Geweke, J. (1993). Bayesian treatment of the independent Student-t
  linear model. Journal of Applied Econometrics 8(S1), S19-S40.
- Chib, S. (1992). Bayes inference in the tobit censored regression
  model. Journal of Econometrics 51(1-2), 79-99.
- Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
  Bayesian additive regression trees. Annals of Applied Statistics 4(1),
  266-298.
- Linero, A. R. (2018). Bayesian regression trees for high-dimensional
  prediction and variable selection. Journal of the American Statistical
  Association 113(522), 626-636.
- Linero, A. R. and Yang, Y. (2018). Bayesian regression tree ensembles
  that adapt to smoothness and sparsity. Journal of the Royal
  Statistical Society Series B 80(5), 1087-1110.
- Polson, N. G., Scott, J. G. and Windle, J. (2013). Bayesian inference
  for logistic models using Polya-Gamma latent variables. Journal of the
  American Statistical Association 108(504), 1339-1349.
- Pratola, M. T., Chipman, H. A., George, E. I. and McCulloch, R. E.
  (2020). Heteroscedastic BART via multiplicative regression trees.
  Journal of Computational and Graphical Statistics 29(2), 405-417.
- Prado, E. B., Moral, R. A. and Parnell, A. C. (2021). Bayesian
  additive regression trees with model trees. Statistics and Computing
  31, 20.
- Robert, C. P. (1995). Simulation of truncated normal variables.
  Statistics and Computing 5, 121-125.
- Sparapani, R., Spanbauer, C. and McCulloch, R. (2021). Nonparametric
  machine learning and efficient computation with Bayesian additive
  regression trees: the BART R package. Journal of Statistical Software
  97(1), 1-66.
- Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
  Voronoi tessellations. Journal of Computational and Graphical
  Statistics 34(3), 859-871.
- Tobin, J. (1958). Estimation of relationships for limited dependent
  variables. Econometrica 26(1), 24-36.
- Wei, L. J. (1992). The accelerated failure time model: a useful
  alternative to the Cox regression model in survival analysis.
  Statistics in Medicine 11(14-15), 1871-1879.
