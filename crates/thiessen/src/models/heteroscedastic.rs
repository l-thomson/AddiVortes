//! The heteroscedastic model, [`Model::Heteroscedastic`] (H-AddiVortes;
//! the structure is that of HBART, Pratola, Chipman, George and McCulloch
//! 2020):
//!
//! ```text
//! y_i = f(x_i) + s(x_i) e_i,   e_i ~ N(0, 1),
//! f(x) = sum_{j=1}^m g(x; T_j, M_j),   s^2(x) = prod_{j=1}^{m'} v(x; T'_j, V_j),
//! ```
//!
//! v(x; T', V) the cell value of the cell of variance tessellation T' that
//! x falls in. The mean ensemble is that of the Gaussian model with
//! per-observation precision 1 / s^2(x_i); the variance ensemble has
//! inverse-gamma cell values and a multiplicative backfit. One sweep
//! updates the variance ensemble given the residuals y - f, then the mean
//! ensemble; this is the order of the authors' code, HBART sweeping mean
//! then variance, both valid Gibbs orders.
//!
//! # Priors
//!
//! Mean cells mu ~ N(0, sigma_mu^2), sigma_mu = 0.5 / (k sqrt m) on the
//! response scaled to [-0.5, 0.5], as in the Gaussian model. Each variance
//! cell v ~ Inv-Gamma(nu' / 2, nu' lambda' / 2) with
//!
//! ```text
//! lambda' = lambda^(1 / m'),   nu' = 2 / (1 - (1 - 2 / nu)^(1 / m')),
//! ```
//!
//! lambda calibrated as for the Gaussian model (Pr(sigma < sigma_hat) =
//! q). These make the prior mean of the product s^2(x), which is
//! (nu' lambda' / (nu' - 2))^m', equal to nu lambda / (nu - 2), the prior
//! mean of the Gaussian model's sigma^2; without the matching, a product
//! of m' cell values under the unadjusted prior has a mean and a spread
//! that grow with m' (HBART s. 3.2 for the same construction with trees).
//! nu > 2 is required. The variance ensemble shares lambda_c, omega and
//! sigma_c with the mean ensemble. Every variance cell starts at
//! sigma_hat^(2 / m'), so s^2 starts at sigma_hat^2.
//!
//! # Correspondence
//!
//! With `rbart` (the HBART package): m = `ntree`, m' = `ntreeh`, k = `k`,
//! nu = `overallnu`, sigma_hat = `overallsd`, burn_in = `nskip`, draws =
//! `ndpost`. `rbart` places the (nu', lambda') matching on the same
//! quantities. With the authors' script: m = `m`, m' = `m_var`, nu = `nu`,
//! q = `q`, k = `k`, sigma_c = `sd`, omega = `Omega`, lambda_c =
//! `lambda_rate` (the script defaults to 25; the crate's default is 5 for
//! every model), burn_in = `burn_in`, draws is `max_iter - burn_in`.
//!
//! # Fitted model
//!
//! `predict` is the posterior mean of f(x); `predict_variance` is
//! s_d^2(x) per draw on the caller's scale (the square of `rbart`'s
//! `sdraws`); `prediction_interval` and `log_likelihood` use
//! N(f_d(x), s_d^2(x)); `sigma` is empty, there being no global sigma.
//!
//! [`Model::Heteroscedastic`]: crate::Model::Heteroscedastic

use crate::config::Config;
use crate::data::Data;
use crate::error::Result;
use crate::fitted::Fitted;

/// Fit the heteroscedastic model with the shared sweep schedule.
pub(crate) fn fit(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Fitted> {
    super::run(config, x, y, seed, progress)
}
