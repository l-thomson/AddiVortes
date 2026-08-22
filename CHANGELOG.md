# Changelog

Notable changes to the core crate, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
Versions follow semantic versioning. Patch releases preserve sampled
values for a fixed seed; minor releases may change them and the entry
says "Sampled values changed" with the reason.

## [Unreleased]

### Added

- (experimental) `Metric::Minkowski` with order p >= 1 and
  `Metric::Manhattan`, its p = 1 alias: Minkowski distance on the scaled
  active coordinates, the active columns of one order combined as
  (sum |x_d - c_d|^p)^(2 / p), so p = 2 is Euclidean bit-for-bit. Centre
  coordinates and structural moves are those of Euclidean; only the
  assignment of rows to cells changes. Outside the semver promise
  (docs/experimental.md).
- (experimental) `Metric::Cosine`: 1 minus the cosine similarity of the
  active Cosine coordinates, squared into the key; 1 against a single
  zero vector, 0 when both are zero. The triangle inequality does not
  hold, the [-0.5, 0.5] scaling makes the origin data-dependent, and the
  option is intended for covariates that are directions already. Outside
  the semver promise (docs/experimental.md).
- (experimental) `Metric::Gower` with a per-column kind: the mean, over
  the active Gower columns, of the range-normalised absolute difference
  (numeric, the [-0.5, 0.5] scaling) or the plain mismatch of integer
  level codes (categorical, levels learnt at fit, uniform centre
  coordinates), squared into the key. Missing-value weighting is not
  implemented. Outside the semver promise (docs/experimental.md).
- (experimental) `Metric::Mahalanobis` with `GeometryParams::precision`,
  a user-supplied row-major p x p matrix over the encoded design, checked
  at fit (symmetric, positive definite). The active-subspace distance
  uses the principal submatrix on the active columns, which is not the
  conditional precision; the identity matrix reproduces Euclidean
  bit-for-bit. Outside the semver promise (docs/experimental.md).
- `Fitted::pool`, the kept draws of chains of the same model and data as
  one fitted model in chain order, with `Error::MismatchedChains` for
  chains that disagree. Chain seeds come from `chain_seed`; the pooled
  in-sample RMSE is that of the pooled posterior mean.
- `fit_with_progress`, `fit` with a callback taking the number of
  completed sweeps and `burn_in + draws * thinning`. The draws for a
  fixed seed are those of `fit`.
- `VERSION`, the crate version as the bindings report it.
- `Metric` and `Config::metric`: the metric of each covariate column,
  Euclidean (default) or spherical, the great-circle distance of CRAN
  AddiVortes `metric = "S"` with `members` as the sphere label. Spherical
  columns are radians, unscaled, with the upstream centre-coordinate
  law; an upstream comparison fixture on a sphere joins the suite. The
  Euclidean chain for a fixed seed is unchanged.
- `Metric::Categorical`: integer level codes with the Eskin mismatch
  weight 2 / n^2 of CRAN AddiVortes `metric = "C"`, uniform centre
  coordinates over the training levels, `Error::InvalidCategoryCode` for
  a non-integer value; the levels are stored on `Fitted`.
- Cargo feature `experimental` for components and models beyond the
  published method, outside the semver promise; `Error::RequiresFeature`
  and `FromStr` for `Model`; the stability contract in the crate-root
  documentation and `docs/experimental.md`.

- `Model::{Gaussian, Probit, Heteroscedastic}` on `Config` and `Fitted`,
  with the per-model fields `Config::offset` (probit) and `Config::m_var`
  (heteroscedastic). The sampler composes a mean ensemble with one of
  three noise models: the global sigma^2 draw, the Albert-Chib latent
  refresh with unit variance, or a multiplicative ensemble of
  inverse-gamma variance tessellations. The Gaussian chain for a fixed
  seed is unchanged.
- `Fitted::model`, `Fitted::predict_latent`, `Fitted::predict_variance`;
  `Sampler::variance_tessellations`, `Sampler::noise_variances`;
  `Posterior::variance_tessellations`; `Error::InvalidLabel` and
  `Error::NotApplicable`. Saved Gaussian models from earlier builds load
  unchanged.
- Probit model (`Model::Probit`): Albert-Chib augmentation, sigma_mu =
  3 / (k sqrt m) on the latent scale, offset Phi^-1(ybar) by default;
  known-answer test by quadrature, SBC and Geweke tests at two sizes, a
  simulation-recovery test and a fixed-seed snapshot; `docs/models.md`
  with the model statements and parameter correspondence tables;
  `benchmarks/upstream/binary_variant.R` for the informational comparison
  against the authors' script.
- Heteroscedastic model (`Model::Heteroscedastic`): a multiplicative
  ensemble of `m_var` inverse-gamma variance tessellations with the
  (nu', lambda') prior matching of HBART; SBC and Geweke tests at two
  sizes, a broken-sampler fixture dropping the inverse-gamma cell
  normaliser, a prior-matching test, a simulation-recovery test and a
  fixed-seed snapshot; `benchmarks/upstream/heteroscedastic_variant.R`
  for the informational comparison against the authors' script. The
  Geweke tests thin at 45 for every model.

- Gaussian AddiVortes model: `fit`, the `Sampler` step API, and `Fitted`
  with prediction, interval, likelihood and ensemble-summary methods.
- `Config::prior_only`: sampling with the likelihood switched off, so
  `predict` gives prior predictive draws.
- `Sampler::pinned_prior`: a sampler whose prior is fixed by the caller,
  for calibration tests; `Sampler::lambda` reports the sigma^2 prior
  scale in force.
- Simulation-based calibration and Geweke joint-distribution tests for
  the Gaussian model at two sizes: small in `cargo test`, full in the
  nightly suite with an R evaluation (SBC package ECDF difference
  bands).
- Broken-sampler fixtures under `cfg(test)`, each rejected by the small
  SBC configuration.
- `Config::paper()`: the paper's settings, lambda_c = 25. The default
  lambda_c is 5 for every model, following CRAN AddiVortes >= 0.6.8.
- Upstream comparison: posterior summaries against CRAN AddiVortes 0.6.9
  on fixed datasets, within 4 combined Monte Carlo standard errors.

### Changed

- Breaking: the `Model` enum is gone. The outcome layer is the whole
  selection surface: `Fitted::model_name()` returns "gaussian",
  "probit" or "heteroscedastic", `Fitted::outcome()` and
  `Fitted::has_variance_ensemble()` carry the identity, and
  `Error::NotApplicable` names the model as a string. A saved flat
  configuration fails to load with an error naming the replacement
  groups. Sampled values are unchanged for a fixed seed.
- Breaking: the configuration is four groups. `outcome` names the
  observation model and carries its own parameters and the sigma^2
  prior (`Outcome::Gaussian(GaussianParams { nu, q })`,
  `Outcome::Probit(ProbitParams { offset })`); `mean_params` and
  `variance_params` are one term-group struct instantiated per slot
  (`num_tessellations`, `k`, `lambda_c`, `geometry` with `metric` and
  `sigma_c`, `structure` with `omega`, `cell`); `general_params` is the
  sweep schedule. H-AddiVortes is `variance_params.num_tessellations`
  above 0 in place of a `Model` variant, defaulting to the paper's 40
  through `with_model`. Validity is derived from the outcome's scale
  mode: a variance ensemble under probit is rejected at config
  assembly, naming identification. The `with_*` setters remain and
  write into the groups; geometry and structure setters write both
  slots, which must agree while per-ensemble geometry awaits its
  identification argument. Sampled values are unchanged for a fixed
  seed.
- `Config` is non-exhaustive; construct it with `Config::new` and the
  `with_*` setters.
- A response lying exactly on a least-squares fit no longer degenerates
  the sigma^2 prior; sigma_hat falls back to the response standard
  deviation.
- Reproducibility contract in the crate-root documentation, with
  determinism and fixed-seed snapshot tests.
- CI pipeline, nightly statistical suite and branch protection.
- Input-data contract in the crate-root documentation; a design matrix
  with no columns is rejected with `Error::NoFeatures`.
- The crate readme is the repository README, so `cargo package` finds it.
