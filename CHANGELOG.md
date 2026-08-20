# Changelog

Notable changes to the core crate, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.
Versions follow semantic versioning. Patch releases preserve sampled
values for a fixed seed; minor releases may change them and the entry
says "Sampled values changed" with the reason.

## [Unreleased]

### Added

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
