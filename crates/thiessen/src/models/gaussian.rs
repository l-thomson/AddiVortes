//! The Gaussian model of Stone and Gosling (2025):
//!
//! ```text
//! y_i = sum_{j=1}^m g(x_i; T_j, M_j) + e_i,   e_i ~ N(0, sigma^2),
//! ```
//!
//! g(x; T, M) the cell mean of the cell of tessellation T that x falls in.
//! Priors (s. 2.3): cell means mu ~ N(0, sigma_mu^2), sigma_mu = 0.5 / (k sqrt
//! m) on the response scaled to [-0.5, 0.5]; sigma^2 ~ nu lambda / chi^2_nu
//! with lambda set so that Pr(sigma < sigma_hat) = q; cells b - 1 ~
//! Poisson(lambda_c); active covariates d - 1 ~ Binomial(p - 1, omega / p);
//! centre coordinates N(0, sigma_c^2) in scaled space. Parameters
//! correspond to the BART package as m = ntree, nu = sigdf, q = sigquant,
//! k = k, burn_in = nskip, draws = ndpost, thinning = keepevery (Sparapani,
//! Spanbauer and McCulloch 2021).

use crate::config::Config;
use crate::data::Data;
use crate::error::Result;
use crate::fitted::Fitted;
use crate::sampler::Sampler;

/// Fit the Gaussian model: validate, run `burn_in` sweeps, then keep every
/// `thinning`-th of the next `draws * thinning` sweeps.
///
/// # Arguments
///
/// `x` is n by p, `y` has n rows, both on the caller's scale; `seed` keys
/// the chain RNG.
///
/// # Errors
///
/// [`Sampler::new`].
pub fn fit(config: &Config, x: &Data, y: &[f64], seed: u64) -> Result<Fitted> {
    let mut sampler = Sampler::new(config, x, y, seed)?;
    for _ in 0..config.burn_in {
        sampler.step();
    }
    for _ in 0..config.draws {
        for _ in 0..config.thinning {
            sampler.step();
        }
        sampler.keep();
    }
    sampler.finish()
}
