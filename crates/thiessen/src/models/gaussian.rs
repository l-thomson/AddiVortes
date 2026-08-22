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
use crate::maths;
use crate::outcome::{OutcomeModel, RequiredData, Sigma2Mode};
use crate::rng::Rng;

/// Fit the Gaussian model with the shared sweep schedule.
pub(crate) fn fit(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Fitted> {
    super::run(config, x, y, seed, progress)
}

/// The Gaussian outcome behind the [`OutcomeModel`] contract: the response
/// is observed, so the working response is y unchanged, the weights are
/// unit, and sigma^2 is sampled by the kernel from its inverse-gamma
/// conditional. The model carries no state of its own.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct GaussianOutcome;

impl GaussianOutcome {
    /// The half-width of the cell-mean prior on the response scaled to
    /// [-0.5, 0.5]: sigma_mu = 0.5 / (k sqrt m) (Stone and Gosling 2025,
    /// s. 2.3.2; Chipman, George and McCulloch 2010, s. 2.2.3).
    pub(crate) const CELL_PRIOR_HALF_WIDTH: f64 = 0.5;
}

impl OutcomeModel for GaussianOutcome {
    fn required_data(&self) -> RequiredData {
        RequiredData::Continuous
    }

    fn init(&mut self, _y: &[f64]) {}

    fn draw_extra(&mut self, _rng: &mut Rng) {}

    fn working_response(&mut self, _total: &[f64], _y: &mut [f64], _rng: &mut Rng) {}

    fn weights(&self) -> Option<&[f64]> {
        None
    }

    fn sigma2_mode(&self) -> Sigma2Mode {
        Sigma2Mode::Sampled
    }

    fn predictive_quantile(&self, mean: f64, sd: f64, p: f64) -> Option<f64> {
        Some(mean + sd * maths::normal_quantile(p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gaussian_outcome_answers_the_contract() {
        let mut outcome = GaussianOutcome;
        let mut rng = crate::rng::chain_rng(3);
        outcome.init(&[0.1, -0.1]);
        outcome.draw_extra(&mut rng);
        let mut y = vec![0.1, -0.1];
        outcome.working_response(&[0.4, 0.4], &mut y, &mut rng);
        assert_eq!(y, vec![0.1, -0.1]);
        assert_eq!(outcome.weights(), None);
        assert_eq!(outcome.sigma2_mode(), Sigma2Mode::Sampled);
        assert_eq!(outcome.required_data(), RequiredData::Continuous);
        let q = outcome.predictive_quantile(1.0, 2.0, 0.975).unwrap();
        assert!((q - (1.0 + 2.0 * 1.959_963_984_540_054)).abs() < 1e-9);
        let median = outcome.predictive_quantile(1.0, 2.0, 0.5).unwrap();
        assert!((median - 1.0).abs() < 1e-12);
    }
}
