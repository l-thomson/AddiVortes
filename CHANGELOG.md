# Changelog

Notable changes to the core crate, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
Versions follow semantic versioning. Patch releases preserve sampled
values for a fixed seed; minor releases may change them and the entry
says "Sampled values changed" with the reason.

## [Unreleased]

### Added

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
- Upstream comparison: posterior summaries against CRAN AddiVortes 0.6.9
  on fixed datasets, within 4 combined Monte Carlo standard errors.

### Changed

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
