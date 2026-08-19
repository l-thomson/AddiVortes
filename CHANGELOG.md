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
- Reproducibility contract in the crate-root documentation, with
  determinism and fixed-seed snapshot tests.
- CI pipeline, nightly statistical suite and branch protection.
- Input-data contract in the crate-root documentation; a design matrix
  with no columns is rejected with `Error::NoFeatures`.
