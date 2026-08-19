# thiessen

Bayesian additive Voronoi tessellation regression (AddiVortes): a Rust
core with Python and R packages.

> Status: pre-release. Nothing is published to crates.io, PyPI or
> CRAN yet. The name `thiessen` is a working name.

## About

`thiessen` implements the AddiVortes method (Stone and Gosling 2025, JCGS
34(3):859-871) and its published variants, Binary AddiVortes (probit
classification) and H-AddiVortes (heteroscedastic variance). All credit for
the method belongs to its authors; the original R package is
[`AddiVortes`](https://github.com/johnpaulgosling/AddiVortes).

The model is `Y = sum_{j=1..m} g(x | T_j, M_j) + e` with `e ~ N(0, sigma^2)`:
a sum of `m` Voronoi tessellations, each partitioning a random subspace of
the covariates, explored by a Gibbs backfitting sampler with
Metropolis-Hastings moves on the tessellation structure.

## Reproducibility

Same seed, same `thiessen` version and same target triple give identical
draws; the full contract is in the crate-root documentation
([crates/thiessen/src/lib.rs](crates/thiessen/src/lib.rs)).

## Licence

Licensed under either of

- Apache License, Version 2.0 ([`LICENSE-APACHE`](LICENSE-APACHE))
- MIT licence ([`LICENSE-MIT`](LICENSE-MIT))

at your option.

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
licence, shall be dual licensed as above, without any additional terms or
conditions.
