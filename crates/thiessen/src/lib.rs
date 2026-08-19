//! Bayesian additive Voronoi tessellations (AddiVortes; Stone and Gosling
//! 2025, JCGS 34(3):859-871): the Gaussian model, a Gibbs sampler with a
//! step API, `fit` and `predict`.
//!
//! # Example
//!
//! ```
//! use thiessen::{fit, Config, Data};
//!
//! # fn main() -> thiessen::Result<()> {
//! let n = 30;
//! let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
//! let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
//! let x = Data::new(xs, n, 1)?;
//!
//! let config = Config::new().with_m(10).with_burn_in(20).with_draws(30);
//! let model = fit(&config, &x, &y, 42)?;
//! let predictions = model.predict(&x)?;
//! assert_eq!(predictions.len(), n);
//! # Ok(())
//! # }
//! ```
//!
//! # Reproducibility
//!
//! The same seed, crate version and target triple give identical draws;
//! draws do not depend on thread count. Transcendental functions go through
//! `libm`, so the reference target `x86_64-unknown-linux-gnu` does not
//! drift with system libc releases. Distribution sampling is pinned to a
//! minor series of `rand_distr` and follows the value-stability policy of
//! `rand` (Rust Rand Book, Reproducibility chapter). Across targets results
//! are statistically equivalent and are compared by posterior summaries,
//! never draw by draw. Patch releases preserve sampled values for a fixed
//! seed; minor releases may change them and the changelog entry says
//! "Sampled values changed" with the reason. Fixed-seed chains are stored
//! under `tests/snapshots/` and checked bit-exact on the reference target;
//! other targets check posterior summaries against the stored chain within
//! Monte Carlo error.
//!
//! # Input data
//!
//! `x` is a numeric matrix; the response is numeric. Missing or non-finite
//! values, a constant response, a constant column, fewer than two rows, or a
//! row-count mismatch are errors, never repaired. Categorical covariates are
//! encoded by the caller (the Python and R packages do this) before they
//! reach the crate.
#![deny(missing_docs)]
#![forbid(unsafe_code)]
#![deny(rustdoc::broken_intra_doc_links)]

mod cells;
mod config;
mod data;
mod error;
mod fitted;
mod maths;
mod models;
mod moves;
mod rng;
mod sampler;
mod scaler;
mod tessellation;

pub use config::Config;
pub use data::{Data, Warning};
pub use error::{Error, Result};
pub use fitted::{Fitted, Interval, Posterior};
pub use models::gaussian::fit;
pub use rng::chain_seed;
pub use sampler::Sampler;
pub use scaler::Scaler;
pub use tessellation::Tessellation;
