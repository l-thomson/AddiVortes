//! The Gaussian model of Stone and Gosling (2025), [`Model::Gaussian`]:
//!
//! ```text
//! y_i = sum_{j=1}^m g(x_i; T_j, M_j) + e_i,   e_i ~ N(0, sigma^2),
//! ```
//!
//! g(x; T, M) the cell mean of the cell of tessellation T that x falls in.
//!
//! # Priors
//!
//! Section 2.3 of the paper: cell means mu ~ N(0, sigma_mu^2), sigma_mu =
//! 0.5 / (k sqrt m) on the response scaled to [-0.5, 0.5]; sigma^2 ~ nu
//! lambda / chi^2_nu with lambda set so that Pr(sigma < sigma_hat) = q,
//! sigma_hat the residual standard deviation of a least-squares fit;
//! cells b - 1 ~ Poisson(lambda_c); active covariates d - 1 ~ Binomial(p -
//! 1, omega / p); centre coordinates N(0, sigma_c^2) in scaled space.
//! Nothing is fixed: every parameter above is sampled.
//!
//! # Correspondence
//!
//! With the BART package (Sparapani, Spanbauer and McCulloch 2021):
//! m = `ntree`, nu = `sigdf`, q = `sigquant`, k = `k`, burn_in = `nskip`,
//! draws = `ndpost`, thinning = `keepevery`. With CRAN AddiVortes:
//! m = `m`, nu = `nu`, q = `q`, k = `k`, sigma_c = `sd`, omega = `Omega`,
//! lambda_c = `LambdaRate`, burn_in = `mcmcBurnIn`, and draws is
//! `totalMCMCIter - mcmcBurnIn`.
//!
//! # Fitted model
//!
//! `predict` is the posterior mean of f(x); `predict_variance` is
//! sigma_d^2 per draw, constant across rows; `prediction_interval` and
//! `log_likelihood` use N(f_d(x), sigma_d^2); `sigma` is sigma_d on the
//! caller's scale.
//!
//! [`Model::Gaussian`]: crate::Model::Gaussian

use crate::config::Config;
use crate::data::Data;
use crate::error::Result;
use crate::fitted::Fitted;

/// Fit the Gaussian model with the shared sweep schedule.
pub(crate) fn fit(config: &Config, x: &Data, y: &[f64], seed: u64) -> Result<Fitted> {
    super::run(config, x, y, seed)
}
