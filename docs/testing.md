# Testing strategy

The suite is layered. Each layer detects a class of defect the layers
before it cannot; none proves correctness on its own.

## Layers

### Unit and property tests

In `src/*`, run by `cargo test`. Closed-form values
for the move selection table and its boundary folding, the count-prior
ratios, cell-statistic accumulation, and equality of every incremental
update with a full recomputation: assignments after each move kind, the
running ensemble fit, and log-marginal differences against a dense
Cholesky evaluation. They detect coding errors in a component; they say
nothing about a mispriced but self-consistent sampler.

### Determinism and snapshots

In `tests/determinism.rs` and `tests/snapshot.rs`. The same seed, crate version and target triple
give identical draws; a fixed-seed chain is stored under
`tests/snapshots/` and checked bit-exact on `x86_64-unknown-linux-gnu`,
with other targets checking posterior summaries within Monte Carlo
error. They detect unintended changes to sampled values, including
platform drift; they say nothing about statistical correctness. The
regeneration procedure is in [CONTRIBUTING.md](../CONTRIBUTING.md).

### Known answers

In `tests/known_answer.rs` and unit tests. Cases with
an independent answer: the conjugate Gibbs chain on a fixed tessellation
against quadrature over the exact posterior; prior-only draws against an
independent rejection sampler of the truncated structural prior
(two-sample chi-squared, alpha 0.001); degenerate inputs against the
documented error or behaviour. They detect mispriced conditionals; they
do not exercise the structural moves against data.

### Upstream comparison

In `tests/upstream.rs`. Posterior summaries on
the Friedman benchmark (n = 200, p = 10) and the `attitude` dataset
against CRAN AddiVortes 0.6.9, every summary within
4 sqrt(mcse_upstream^2 + mcse_ours^2); fixtures regenerate only through
the renv-pinned script in `benchmarks/upstream/`. It detects divergence
from the published implementation; it cannot detect a defect shared with
it.

### Calibration

In `tests/calibration.rs`. Simulation-based calibration
(Talts et al. 2018; Modrák et al. 2025) and the Geweke (2004)
joint-distribution test, run under the pinned prior
(`Sampler::pinned_prior`), which fixes lambda and the scaling so the
prior is not a function of the data. Small configurations run in
`cargo test` with chi-squared and Kolmogorov-Smirnov gates at family
alpha 0.01, Bonferroni-split across the quantities; full configurations
run nightly and are also evaluated in R with the SBC package's rank ECDF
difference bands (Säilynoja, Bürkner and Vehtari 2022). These detect
wrong-but-self-consistent kernels; a pass at these sizes is evidence,
not proof.

### Tests of tests

In `src/broken.rs`, with `cargo-mutants` in the nightly suite. Two
mispriced acceptance ratios under `cfg(test)` (an inflated add-centre
ratio, and the add-dimension selection ratio without its reverse-bound
folding, the defect upstream corrected in 0.6.8) must each be rejected
by the small SBC gate. The nightly mutation report is informational.

### Binding contracts

Once the Python and R packages
exist (in development), each asserts the same seed gives the same draws through the
binding as through the core, and that encodings match the documented
contract.

## Numbers

Every size, alpha and critical value is stated next to the test that
uses it, in the source. Current gate sizes: SBC small 160 simulations,
19 kept draws, chi-squared over 20 rank bins; Geweke small 2000
marginal-conditional draws against 800 successive-conditional keeps at
thinning 15; upstream comparison k = 4 (a two-sided 6e-5 level per
summary); broken-sampler SBC 400 simulations.

## Running the full suite locally

    cargo nextest run --locked --run-ignored all

The full calibration tests write ranks and samples under
`target/calibration`; render the plots with

    Rscript benchmarks/calibration/evaluate.R target/calibration

as the nightly `calibration` job does.

## References

- Talts, S., Betancourt, M., Simpson, D., Vehtari, A. and Gelman, A.
  (2018). Validating Bayesian inference algorithms with simulation-based
  calibration. arXiv:1804.06788.
- Modrák, M., Moon, A. H., Kim, S., Bürkner, P., Huurre, N.,
  Faltejsková, K., Gelman, A. and Vehtari, A. (2025). Simulation-based
  calibration checking for Bayesian computation: the choice of test
  quantities shapes sensitivity. Bayesian Analysis 20(2).
- Geweke, J. (2004). Getting it right: joint distribution tests of
  posterior simulators. Journal of the American Statistical Association
  99(467), 799-804.
- Säilynoja, T., Bürkner, P. and Vehtari, A. (2022). Graphical test for
  discrete uniformity and its applications in goodness-of-fit evaluation
  and multiple sample comparison. Statistics and Computing 32, 32.
- Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
  Voronoi tessellations. Journal of Computational and Graphical
  Statistics 34(3), 859-871.
