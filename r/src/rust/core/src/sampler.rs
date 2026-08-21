//! The Gibbs sampler with Metropolis-Hastings structural moves (Stone and
//! Gosling 2025, Algorithm 1): one sweep per `step`, the caller's loop
//! deciding burn-in and thinning through `keep`, and `finish` producing the
//! fitted model.
//!
//! The sampler composes a mean ensemble (Gaussian cell means, additive)
//! with one of three noise models, chosen by [`Config::model`]: a global
//! sigma^2 drawn from its inverse-gamma conditional (Gaussian model); a
//! fixed unit variance with the Albert and Chib (1993) latent response
//! refreshed before each sweep (probit model); or a variance ensemble
//! (inverse-gamma cell values, multiplicative) whose product at each
//! observation is its variance, updated before the mean sweep
//! (heteroscedastic model; the sweep order of the authors' code, HBART
//! sweeping mean then variance).

use crate::cells::{GaussianCells, InverseGammaCells};
use crate::config::Config;
use crate::data::{self, Data, Warning};
use crate::ensemble::Ensemble;
use crate::error::{Error, Result};
use crate::fitted::{Fitted, Posterior};
use crate::geometry::Geometry;
use crate::maths;
use crate::model::Model;
use crate::moves::Prior;
use crate::rng::{self, Rng};
use crate::scaler::{self, Scaler};
use crate::tessellation::Tessellation;

/// The noise model in force.
#[derive(Debug, Clone)]
enum Noise {
    /// One global variance, scaled space.
    Gaussian { sigma_sq: f64 },
    /// Unit latent variance; the labels in {0, 1} and the offset c of
    /// P(y = 1 | x) = Phi(c + f(x)).
    Unit { labels: Vec<f64>, offset: f64 },
    /// The variance ensemble, whose total at each observation is s^2(x_i).
    Ensemble(Box<Ensemble<InverseGammaCells>>),
}

/// The sampler state for one chain. Construct with [`Sampler::new`], advance
/// with [`step`](Sampler::step), record draws with [`keep`](Sampler::keep),
/// and close with [`finish`](Sampler::finish).
///
/// The response is on the caller's scale through the affine map frozen at
/// construction; [`set_response`](Sampler::set_response) and
/// [`fitted_values`](Sampler::fitted_values) speak that scale. Under the
/// probit model the response is the labels and the fitted values are the
/// latent mean c + f(x). The sampler owns its RNG, seeded at construction;
/// a caller's loop consumes none of it.
#[derive(Debug, Clone)]
pub struct Sampler {
    rng: Rng,
    config: Config,
    scaler: Scaler,
    warnings: Vec<Warning>,
    /// Scaled design.
    x: Data,
    /// Scaled working response: the scaled y, or under the probit model the
    /// latent z - c.
    y: Vec<f64>,
    mean: Ensemble<GaussianCells>,
    noise: Noise,
    /// Per-observation precision of the mean ensemble's observations.
    precision: Vec<f64>,
    /// Calibrated scale of the prior sigma^2 ~ nu lambda / chi^2_nu.
    lambda: f64,
    kept: Posterior,
    /// Test-only acceptance defect; `Breakage::None` on every constructed
    /// sampler.
    #[cfg(test)]
    pub(crate) breakage: crate::broken::Breakage,
}

impl Sampler {
    /// A sampler over raw data, its RNG seeded from `seed`.
    ///
    /// # Errors
    ///
    /// [`Config::validate`], then the data checks: row counts, at least two
    /// observations, finite values, non-constant response and columns,
    /// omega <= p, `metric` naming every column with each sphere's
    /// longitude last, `InvalidCategoryCode` for a non-integer value in a
    /// categorical column, and under the probit model `InvalidLabel` for
    /// a response value outside {0, 1}.
    pub fn new(config: &Config, x: &Data, y: &[f64], seed: u64) -> Result<Self> {
        Self::build(config, x, y, seed, None)
    }

    /// A sampler whose prior is fixed by the caller: `x` and `y` are taken
    /// as already scaled, the affine maps are identity, and `lambda` is
    /// given rather than calibrated, so the prior does not depend on the
    /// data. Calibration tests (SBC, Geweke test) require this
    /// constructor; under [`Sampler::new`] the y-scaling and the lambda
    /// calibration make the prior a function of the data, which
    /// invalidates a joint-distribution test. Under the probit model
    /// `lambda` is not used; `offset` must then be set on the
    /// configuration, since Phi^-1(ybar) is a function of the data.
    ///
    /// # Errors
    ///
    /// [`Sampler::new`], `InvalidHyperparameter` for a non-positive
    /// `lambda`, and for a probit configuration without `offset`.
    pub fn pinned_prior(
        config: &Config,
        x: &Data,
        y: &[f64],
        lambda: f64,
        seed: u64,
    ) -> Result<Self> {
        if !(lambda.is_finite() && lambda > 0.0) {
            return Err(crate::error::invalid(
                "lambda",
                format!("must be finite and positive, got {lambda}"),
            ));
        }
        if config.model == Model::Probit && config.offset.is_none() {
            return Err(crate::error::invalid(
                "offset",
                "must be set under the pinned prior of the probit model",
            ));
        }
        Self::build(config, x, y, seed, Some(lambda))
    }

    fn build(
        config: &Config,
        x: &Data,
        y: &[f64],
        seed: u64,
        pinned_lambda: Option<f64>,
    ) -> Result<Self> {
        config.validate()?;
        let p = x.n_cols();
        let omega = config.omega_for(p);
        data::validate_fit(x, y, omega)?;
        let geometry = Geometry::fit(&config.metric, x)?;
        let laws = geometry.laws(x, config.sigma_c)?;
        if config.model == Model::Probit {
            if let Some(row) = y.iter().position(|&v| v != 0.0 && v != 1.0) {
                return Err(Error::InvalidLabel { row });
            }
        }
        let mut config = config.clone();
        config.omega = Some(omega);
        let warnings = data::fit_warnings(x);
        let n = x.n_rows();

        // Response scaling, lambda and the initial sigma^2 at the model
        // boundary: the probit model keeps the response unscaled.
        let (scaler, x_scaled, y_scaled, lambda, sigma_sq) = match (config.model, pinned_lambda) {
            (Model::Probit, Some(lambda)) => {
                (Scaler::identity(p), x.clone(), y.to_vec(), lambda, 1.0)
            }
            (Model::Probit, None) => {
                let (scaler, x_scaled) = Scaler::fit_x(x, &geometry);
                (scaler, x_scaled, y.to_vec(), 1.0, 1.0)
            }
            (_, Some(lambda)) => (Scaler::identity(p), x.clone(), y.to_vec(), lambda, lambda),
            (_, None) => {
                let (scaler, x_scaled, y_scaled) = Scaler::fit(x, y, &geometry);
                let sigma_hat = scaler::sigma_hat(&x_scaled, &y_scaled);
                let lambda = scaler::calibrate_lambda(config.nu, config.q, sigma_hat);
                (scaler, x_scaled, y_scaled, lambda, sigma_hat * sigma_hat)
            }
        };
        let sigma_mu_sq = match config.model {
            Model::Probit => scaler::probit_sigma_mu_sq(config.k, config.m),
            Model::Gaussian | Model::Heteroscedastic => scaler::sigma_mu_sq(config.k, config.m),
        };
        let prior = Prior {
            p,
            omega,
            lambda_c: config.lambda_c,
            geometry,
            laws,
        };
        let mut rng = rng::chain_rng(seed);
        let mean_y = y_scaled.iter().sum::<f64>() / n as f64;

        // Initial state: m single-cell tessellations on one covariate each,
        // every cell mean ybar / m so the ensemble fit starts at ybar; under
        // the probit model f starts at 0 and the offset carries the mean.
        let (cell_value, total) = match config.model {
            Model::Probit => (0.0, 0.0),
            Model::Gaussian | Model::Heteroscedastic => (mean_y / config.m as f64, mean_y),
        };
        let mean = Ensemble::new(
            GaussianCells { sigma_mu_sq },
            prior.clone(),
            &x_scaled,
            config.m,
            cell_value,
            total,
            &mut rng,
        );
        let noise = match config.model {
            Model::Gaussian => Noise::Gaussian { sigma_sq },
            Model::Probit => {
                let offset = config
                    .offset
                    .unwrap_or_else(|| maths::normal_quantile(mean_y));
                config.offset = Some(offset);
                Noise::Unit {
                    labels: y_scaled.clone(),
                    offset,
                }
            }
            Model::Heteroscedastic => {
                let (nu, lambda_prime) =
                    scaler::variance_cell_prior(config.nu, lambda, config.m_var);
                // Every variance cell starts at sigma^2 ^ (1 / m') so the
                // product starts at the initial sigma^2.
                Noise::Ensemble(Box::new(Ensemble::new(
                    InverseGammaCells {
                        nu,
                        lambda: lambda_prime,
                        prior_only: config.prior_only,
                    },
                    prior,
                    &x_scaled,
                    config.m_var,
                    libm::pow(sigma_sq, 1.0 / config.m_var as f64),
                    sigma_sq,
                    &mut rng,
                )))
            }
        };
        // Zero precision removes the likelihood from every conditional:
        // the integrated-likelihood terms of the acceptance ratio vanish
        // and the cell means are drawn from N(0, sigma_mu^2).
        let precision = if config.prior_only {
            vec![0.0; n]
        } else {
            vec![1.0 / sigma_sq; n]
        };
        let y = match config.model {
            Model::Probit => vec![0.0; n],
            Model::Gaussian | Model::Heteroscedastic => y_scaled,
        };

        Ok(Self {
            rng,
            config,
            scaler,
            warnings,
            x: x_scaled,
            y,
            mean,
            noise,
            precision,
            lambda,
            kept: Posterior::empty(),
            #[cfg(test)]
            breakage: crate::broken::Breakage::None,
        })
    }

    /// One sweep: the noise update (sigma^2 | rest; the latent response |
    /// rest; or the variance ensemble | rest), then for each mean
    /// tessellation in turn a structural move, accept or reject, and the
    /// cell means | rest.
    pub fn step(&mut self) {
        self.update_noise(true);
        self.mean.sweep(
            &self.x,
            &self.y,
            &self.precision,
            &mut self.rng,
            #[cfg(test)]
            self.breakage,
        );
    }

    /// The noise model's update; `structural` enables the variance
    /// ensemble's structural moves, which the conjugate sweep of the
    /// known-answer tests disables.
    fn update_noise(&mut self, structural: bool) {
        let prior_only = self.config.prior_only;
        match &mut self.noise {
            // sigma^2 | y, F ~ Inv-Gamma((nu + n) / 2, (nu lambda + sum r_i^2) / 2)
            // with r = y - F; under prior-only sampling the prior
            // Inv-Gamma(nu / 2, nu lambda / 2).
            Noise::Gaussian { sigma_sq } => {
                let nu = self.config.nu;
                let (shape, scale) = if prior_only {
                    (0.5 * nu, 2.0 / (nu * self.lambda))
                } else {
                    let rss: f64 = self
                        .y
                        .iter()
                        .zip(self.mean.total())
                        .map(|(y, f)| (y - f) * (y - f))
                        .sum();
                    let n = self.y.len();
                    (0.5 * (nu + n as f64), 2.0 / (nu * self.lambda + rss))
                };
                *sigma_sq = 1.0 / rng::gamma(shape, scale, &mut self.rng);
                if !prior_only {
                    let precision = 1.0 / *sigma_sq;
                    self.precision.iter_mut().for_each(|w| *w = precision);
                }
            }
            // z_i ~ N(c + f(x_i), 1) truncated to z_i > 0 when y_i = 1 and
            // z_i < 0 when y_i = 0; the working response is z_i - c.
            Noise::Unit { labels, offset } => {
                if prior_only {
                    return;
                }
                for ((slot, &label), &f) in
                    self.y.iter_mut().zip(labels.iter()).zip(self.mean.total())
                {
                    let mean = f + *offset;
                    let z = if label == 1.0 {
                        mean + rng::truncated_standard_normal_above(-mean, &mut self.rng)
                    } else {
                        mean - rng::truncated_standard_normal_above(mean, &mut self.rng)
                    };
                    *slot = z - *offset;
                }
            }
            // The variance ensemble given the residuals e = y - F, then the
            // precision 1 / s^2(x_i).
            Noise::Ensemble(variance) => {
                let residuals: Vec<f64> = self
                    .y
                    .iter()
                    .zip(self.mean.total())
                    .map(|(y, f)| y - f)
                    .collect();
                if structural {
                    variance.sweep(
                        &self.x,
                        &residuals,
                        &self.precision,
                        &mut self.rng,
                        #[cfg(test)]
                        self.breakage,
                    );
                } else {
                    #[cfg(test)]
                    variance.conjugate_sweep(&residuals, &self.precision, &mut self.rng);
                    #[cfg(not(test))]
                    unreachable!("the conjugate sweep exists only under test");
                }
                if !prior_only {
                    for (w, &s) in self.precision.iter_mut().zip(variance.total()) {
                        *w = 1.0 / s;
                    }
                }
            }
        }
    }

    /// Record the current state as a posterior draw.
    pub fn keep(&mut self) {
        let sigma_sq = match &self.noise {
            Noise::Gaussian { sigma_sq } => Some(*sigma_sq),
            Noise::Unit { .. } | Noise::Ensemble(_) => None,
        };
        let variance = match &self.noise {
            Noise::Ensemble(v) => Some(v.tessellations().to_vec()),
            Noise::Gaussian { .. } | Noise::Unit { .. } => None,
        };
        self.kept
            .push(sigma_sq, self.mean.tessellations().to_vec(), variance);
    }

    /// Number of draws kept so far.
    pub fn n_kept(&self) -> usize {
        self.kept.n_draws()
    }

    /// Replace the response (caller scale; labels in {0, 1} under the
    /// probit model), keeping the tessellations, the cell values and
    /// sigma^2. The next sweep conditions on the new response.
    ///
    /// # Errors
    ///
    /// `RowCountMismatch`, `NonFiniteResponse`, `InvalidLabel`.
    pub fn set_response(&mut self, y: &[f64]) -> Result<()> {
        if y.len() != self.y.len() {
            return Err(Error::RowCountMismatch {
                y_len: y.len(),
                x_rows: self.y.len(),
            });
        }
        if let Some(row) = y.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteResponse { row });
        }
        if let Noise::Unit { labels, .. } = &mut self.noise {
            if let Some(row) = y.iter().position(|&v| v != 0.0 && v != 1.0) {
                return Err(Error::InvalidLabel { row });
            }
            labels.copy_from_slice(y);
            return Ok(());
        }
        for (slot, &v) in self.y.iter_mut().zip(y) {
            *slot = self.scaler.scale_y(v);
        }
        Ok(())
    }

    /// The current mean function at the training rows, caller scale: f(x_i),
    /// or c + f(x_i) under the probit model.
    pub fn fitted_values(&self) -> Vec<f64> {
        let offset = self.offset();
        self.mean
            .total()
            .iter()
            .map(|&f| self.scaler.unscale_y(f) + offset)
            .collect()
    }

    /// The variance of y_i given f at each training row, caller scale:
    /// sigma^2 under the Gaussian model, 1 under the probit model (the
    /// latent scale), s^2(x_i) under the heteroscedastic model.
    pub fn noise_variances(&self) -> Vec<f64> {
        let n = self.y.len();
        let range_sq = self.scaler.y_range() * self.scaler.y_range();
        match &self.noise {
            Noise::Gaussian { sigma_sq } => vec![sigma_sq * range_sq; n],
            Noise::Unit { .. } => vec![1.0; n],
            Noise::Ensemble(v) => v.total().iter().map(|s| s * range_sq).collect(),
        }
    }

    /// The current global sigma^2, scaled space: the Gaussian model's draw,
    /// or 1 under the probit and heteroscedastic models, whose variance is
    /// fixed at 1 on the latent scale and carried by the variance ensemble
    /// respectively.
    pub fn sigma_sq(&self) -> f64 {
        match &self.noise {
            Noise::Gaussian { sigma_sq } => *sigma_sq,
            Noise::Unit { .. } | Noise::Ensemble(_) => 1.0,
        }
    }

    /// lambda of the sigma^2 prior nu lambda / chi^2_nu: calibrated from
    /// the data at construction, or the value given to
    /// [`pinned_prior`](Sampler::pinned_prior). The heteroscedastic model
    /// derives its per-cell lambda' = lambda^(1 / m') from it; the probit
    /// model does not use it.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The current mean tessellations, scaled space.
    pub fn tessellations(&self) -> &[Tessellation] {
        self.mean.tessellations()
    }

    /// The current variance tessellations, scaled space; empty outside the
    /// heteroscedastic model.
    pub fn variance_tessellations(&self) -> &[Tessellation] {
        match &self.noise {
            Noise::Ensemble(v) => v.tessellations(),
            Noise::Gaussian { .. } | Noise::Unit { .. } => &[],
        }
    }

    /// The scaling frozen at construction.
    pub fn scaler(&self) -> &Scaler {
        &self.scaler
    }

    /// The configuration, with omega (and under the probit model the
    /// offset) resolved.
    pub fn config(&self) -> &Config {
        &self.config
    }

    fn offset(&self) -> f64 {
        match &self.noise {
            Noise::Unit { offset, .. } => *offset,
            Noise::Gaussian { .. } | Noise::Ensemble(_) => 0.0,
        }
    }

    /// The fitted model from the kept draws.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter` for `draws` when nothing was kept.
    pub fn finish(self) -> Result<Fitted> {
        if self.kept.n_draws() == 0 {
            return Err(crate::error::invalid("draws", "no draws were kept"));
        }
        let n = self.y.len();
        // The posterior-mean prediction on the response scale: f, or
        // Phi(c + f) under the probit model.
        let offset = self.offset();
        let probit = matches!(self.noise, Noise::Unit { .. });
        let mut mean_prediction = vec![0.0; n];
        let geometry = self.mean.geometry();
        for draw in self.kept.tessellations() {
            for (i, slot) in mean_prediction.iter_mut().enumerate() {
                let row = self.x.row(i);
                let f: f64 = draw.iter().map(|t| t.value_at(row, geometry)).sum();
                *slot += if probit {
                    maths::normal_cdf(f + offset)
                } else {
                    f
                };
            }
        }
        let n_draws = self.kept.n_draws() as f64;
        let range = self.scaler.y_range();
        let target: &[f64] = match &self.noise {
            Noise::Unit { labels, .. } => labels,
            Noise::Gaussian { .. } | Noise::Ensemble(_) => &self.y,
        };
        let in_sample_rmse = (mean_prediction
            .iter()
            .zip(target)
            .map(|(f, y)| {
                let r = (f / n_draws - y) * range;
                r * r
            })
            .sum::<f64>()
            / n as f64)
            .sqrt();
        let categories = self.mean.geometry().categories().to_vec();
        Ok(Fitted::new(
            self.config,
            self.scaler,
            self.kept,
            self.warnings,
            in_sample_rmse,
            categories,
        ))
    }
}

#[cfg(test)]
impl Sampler {
    /// One sweep with the structural moves disabled: the noise update, then
    /// every mean tessellation's cell means | rest. On a fixed tessellation
    /// the chain is the conjugate Gibbs sampler of the known-answer tests.
    pub(crate) fn conjugate_sweep(&mut self) {
        self.update_noise(false);
        self.mean
            .conjugate_sweep(&self.y, &self.precision, &mut self.rng);
    }

    /// Replace mean tessellation `j` by `t`, whose cell means must be zero,
    /// and reset the running fit to zero.
    pub(crate) fn fix_mean_tessellation(&mut self, j: usize, t: Tessellation) {
        debug_assert!(t.mus.iter().all(|&m| m == 0.0));
        let x = self.x.clone();
        self.mean.set_tessellation(j, &x, t, 0.0);
    }

    /// The cell index of every training row under mean tessellation `j`.
    pub(crate) fn mean_cells(&self, j: usize) -> &[usize] {
        &self.mean.assignments()[j].cells
    }

    pub(crate) fn mean_sigma_mu_sq(&self) -> f64 {
        self.mean.family().sigma_mu_sq
    }

    pub(crate) fn variance_ensemble(&self) -> Option<&Ensemble<InverseGammaCells>> {
        match &self.noise {
            Noise::Ensemble(v) => Some(v),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit;
    use crate::tessellation::Assignment;

    fn toy(n: usize) -> (Data, Vec<f64>) {
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
        let x = Data::new(xs, n, 1).unwrap();
        (x, y)
    }

    fn labels(n: usize) -> (Data, Vec<f64>) {
        let (x, y) = toy(n);
        let labels = y.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
        (x, labels)
    }

    fn small() -> Config {
        Config::new().with_m(10).with_burn_in(20).with_draws(30)
    }

    #[test]
    fn same_seed_gives_identical_draws() {
        let (x, y) = toy(30);
        let a = fit(&small(), &x, &y, 42).unwrap();
        let b = fit(&small(), &x, &y, 42).unwrap();
        let c = fit(&small(), &x, &y, 43).unwrap();
        assert_eq!(a.posterior(), b.posterior());
        assert_ne!(a.posterior(), c.posterior());
    }

    #[test]
    fn fit_equals_the_explicit_loop() {
        let (x, y) = toy(30);
        let config = small().with_thinning(2);
        let fitted = fit(&config, &x, &y, 7).unwrap();
        let mut sampler = Sampler::new(&config, &x, &y, 7).unwrap();
        for _ in 0..20 {
            sampler.step();
        }
        for _ in 0..30 {
            sampler.step();
            sampler.step();
            sampler.keep();
        }
        let looped = sampler.finish().unwrap();
        assert_eq!(fitted.posterior(), looped.posterior());
        assert_eq!(fitted.in_sample_rmse(), looped.in_sample_rmse());
    }

    #[test]
    fn running_fit_matches_recomputation() {
        let (x, y) = toy(40);
        let mut sampler = Sampler::new(&small(), &x, &y, 3).unwrap();
        for _ in 0..15 {
            sampler.step();
            let g = sampler.mean.geometry();
            for i in 0..40 {
                let row = sampler.x.row(i);
                let sum: f64 = sampler
                    .tessellations()
                    .iter()
                    .map(|t| t.value_at(row, g))
                    .sum();
                assert!((sum - sampler.mean.total()[i]).abs() < 1e-9);
            }
            for (t, a) in sampler
                .tessellations()
                .iter()
                .zip(sampler.mean.assignments())
            {
                assert_eq!(*a, Assignment::full(&sampler.x, t, g));
            }
        }
    }

    #[test]
    fn running_variance_product_matches_recomputation() {
        let (x, y) = toy(40);
        let config = small().with_model(Model::Heteroscedastic).with_m_var(6);
        let mut sampler = Sampler::new(&config, &x, &y, 3).unwrap();
        for _ in 0..15 {
            sampler.step();
            let variance = sampler.variance_ensemble().unwrap();
            let g = variance.geometry();
            for i in 0..40 {
                let row = sampler.x.row(i);
                let product: f64 = variance
                    .tessellations()
                    .iter()
                    .map(|t| t.value_at(row, g))
                    .product();
                assert!((product - variance.total()[i]).abs() < 1e-9 * product);
                assert!((1.0 / variance.total()[i] - sampler.precision[i]).abs() < 1e-9);
            }
            for (t, a) in variance.tessellations().iter().zip(variance.assignments()) {
                assert_eq!(*a, Assignment::full(&sampler.x, t, g));
            }
        }
    }

    #[test]
    fn set_response_and_fitted_values_on_the_caller_scale() {
        let (x, y) = toy(30);
        let config = small();
        let mut plain = Sampler::new(&config, &x, &y, 1).unwrap();
        let mut same = Sampler::new(&config, &x, &y, 1).unwrap();
        let mut shifted = Sampler::new(&config, &x, &y, 1).unwrap();
        same.set_response(&y).unwrap();
        let up: Vec<f64> = y.iter().map(|v| v + 0.5).collect();
        shifted.set_response(&up).unwrap();
        for _ in 0..100 {
            plain.step();
            same.step();
            shifted.step();
        }
        assert_eq!(plain.fitted_values(), same.fitted_values());
        let mean = |v: Vec<f64>| v.iter().sum::<f64>() / v.len() as f64;
        let (plain_mean, shifted_mean) =
            (mean(plain.fitted_values()), mean(shifted.fitted_values()));
        assert!(
            shifted_mean - plain_mean > 0.3,
            "{plain_mean} {shifted_mean}"
        );
        let mut sampler = plain;
        assert!(matches!(
            sampler.set_response(&[1.0]),
            Err(Error::RowCountMismatch { .. })
        ));
        assert!(matches!(
            sampler.set_response(&vec![f64::NAN; 30]),
            Err(Error::NonFiniteResponse { row: 0 })
        ));
    }

    #[test]
    fn probit_boundary_and_state() {
        let (x, labels) = labels(30);
        let config = small().with_model(Model::Probit);
        let mut bad = labels.clone();
        bad[3] = 0.5;
        assert_eq!(
            Sampler::new(&config, &x, &bad, 1).unwrap_err(),
            Error::InvalidLabel { row: 3 }
        );
        assert!(Sampler::pinned_prior(&config, &x, &labels, 1.0, 1).is_err());
        assert!(
            Sampler::pinned_prior(&config.clone().with_offset(0.0), &x, &labels, 1.0, 1).is_ok()
        );

        let mut sampler = Sampler::new(&config, &x, &labels, 1).unwrap();
        let share = labels.iter().sum::<f64>() / 30.0;
        let offset = sampler.config().offset.unwrap();
        assert!((maths::normal_cdf(offset) - share).abs() < 1e-9);
        // The response is unscaled and f starts at 0, so the fitted values
        // start at the offset.
        assert_eq!(sampler.scaler().y_range(), 1.0);
        assert!(sampler.fitted_values().iter().all(|&v| v == offset));
        assert_eq!(sampler.sigma_sq(), 1.0);
        assert_eq!(sampler.noise_variances(), vec![1.0; 30]);
        assert!(sampler.variance_tessellations().is_empty());
        assert_eq!(
            sampler.mean.family().sigma_mu_sq,
            (3.0 / (3.0 * 10f64.sqrt())).powi(2)
        );
        for _ in 0..50 {
            sampler.step();
        }
        // The latent response has the sign of its label.
        for (z, &label) in sampler.y.iter().zip(&labels) {
            assert_eq!((z + offset > 0.0), label == 1.0);
        }
        assert_eq!(
            sampler.set_response(&vec![2.0; 30]).unwrap_err(),
            Error::InvalidLabel { row: 0 }
        );
        sampler.keep();
        let fitted = sampler.finish().unwrap();
        assert!(fitted.posterior().sigma_sq().is_empty());
        assert!(fitted.in_sample_rmse() < 0.5);
    }

    #[test]
    fn heteroscedastic_state() {
        let (x, y) = toy(30);
        let config = small().with_model(Model::Heteroscedastic).with_m_var(4);
        let mut sampler = Sampler::new(&config, &x, &y, 1).unwrap();
        assert_eq!(sampler.variance_tessellations().len(), 4);
        assert_eq!(sampler.sigma_sq(), 1.0);
        let family = sampler.variance_ensemble().unwrap().family();
        let (nu, lambda) = scaler::variance_cell_prior(6.0, sampler.lambda(), 4);
        assert_eq!((family.nu, family.lambda), (nu, lambda));
        // The product starts at sigma_hat^2 and the precision at its inverse.
        let initial = sampler.noise_variances()[0] / sampler.scaler().y_range().powi(2);
        assert!((1.0 / initial - sampler.precision[0]).abs() < 1e-12);
        for _ in 0..30 {
            sampler.step();
        }
        sampler.keep();
        assert!(sampler
            .noise_variances()
            .iter()
            .all(|v| v.is_finite() && *v > 0.0));
        let fitted = sampler.finish().unwrap();
        assert!(fitted.posterior().sigma_sq().is_empty());
        assert_eq!(fitted.posterior().variance_tessellations()[0].len(), 4);
    }

    #[test]
    fn finish_needs_a_kept_draw() {
        let (x, y) = toy(10);
        let sampler = Sampler::new(&small(), &x, &y, 1).unwrap();
        assert!(sampler.finish().is_err());
    }

    #[test]
    fn omega_is_resolved_on_the_sampler_config() {
        let (x, y) = toy(10);
        let sampler = Sampler::new(&small(), &x, &y, 1).unwrap();
        assert_eq!(sampler.config().omega, Some(1.0));
    }

    #[test]
    fn recovers_a_smooth_function() {
        let n = 200;
        let mut rng = rng::chain_rng(99);
        let xs: Vec<f64> = (0..n).map(|_| rng::uniform(&mut rng)).collect();
        let f = |v: f64| (2.0 * std::f64::consts::PI * v).sin();
        let y: Vec<f64> = xs
            .iter()
            .map(|&v| f(v) + 0.1 * rng::standard_normal(&mut rng))
            .collect();
        let x = Data::new(xs.clone(), n, 1).unwrap();
        let config = Config::new().with_m(50).with_burn_in(200).with_draws(200);
        let fitted = fit(&config, &x, &y, 5).unwrap();
        let pred = fitted.predict(&x).unwrap();
        let rmse = (xs
            .iter()
            .zip(&pred)
            .map(|(&v, &p)| (f(v) - p).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(rmse < 0.15, "rmse {rmse}");
        let sigma = fitted.sigma();
        let mean_sigma = sigma.iter().sum::<f64>() / sigma.len() as f64;
        assert!((0.05..0.2).contains(&mean_sigma), "sigma {mean_sigma}");
    }

    /// Per-cell count, sum and sum of squares of the response under a
    /// fixed assignment.
    struct FixedCells {
        n: Vec<f64>,
        s1: Vec<f64>,
        s2: Vec<f64>,
    }

    impl FixedCells {
        fn accumulate(cells: &[usize], y: &[f64], b: usize) -> Self {
            let mut out = Self {
                n: vec![0.0; b],
                s1: vec![0.0; b],
                s2: vec![0.0; b],
            };
            for (&cell, &v) in cells.iter().zip(y) {
                out.n[cell] += 1.0;
                out.s1[cell] += v;
                out.s2[cell] += v * v;
            }
            out
        }
    }

    /// Posterior means of sigma^2 and of every cell mean for the fixed
    /// tessellation model by quadrature over t = ln sigma^2.
    ///
    /// With mu integrated out cell k contributes, up to constants,
    ///
    /// ```text
    /// ln N(y_k; 0, sigma^2 I + sigma_mu^2 J) =
    ///   -[ (n_k - 1) ln sigma^2 + ln(sigma^2 + n_k sigma_mu^2)
    ///      + (s2_k - sigma_mu^2 s1_k^2 / (sigma^2 + n_k sigma_mu^2)) / sigma^2 ] / 2,
    /// ```
    ///
    /// the prior is Inv-Gamma(nu / 2, nu lambda / 2), and
    /// E[mu_k | sigma^2, y] = sigma_mu^2 s1_k / (sigma^2 + n_k sigma_mu^2).
    fn quadrature_reference(
        stats: &FixedCells,
        nu: f64,
        lambda: f64,
        sigma_mu_sq: f64,
    ) -> (f64, Vec<f64>) {
        let b = stats.n.len();
        let (a, scale) = (0.5 * nu, 0.5 * nu * lambda);
        let steps = 40_000;
        let (lo, hi) = (lambda.ln() - 10.0, lambda.ln() + 10.0);
        let mut log_weights = Vec::with_capacity(steps + 1);
        let mut grid = Vec::with_capacity(steps + 1);
        for i in 0..=steps {
            let t = lo + (hi - lo) * i as f64 / steps as f64;
            let sigma_sq = t.exp();
            let mut lp = -(a + 1.0) * t - scale / sigma_sq + t;
            for k in 0..b {
                let den = sigma_sq + stats.n[k] * sigma_mu_sq;
                lp += -0.5
                    * ((stats.n[k] - 1.0) * t
                        + den.ln()
                        + (stats.s2[k] - sigma_mu_sq * stats.s1[k] * stats.s1[k] / den) / sigma_sq);
            }
            grid.push(sigma_sq);
            log_weights.push(lp);
        }
        let max = log_weights
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_weights.iter().map(|lp| (lp - max).exp()).collect();
        let total: f64 = weights.iter().sum();
        let mean_sigma_sq: f64 = weights.iter().zip(&grid).map(|(w, s)| w * s).sum::<f64>() / total;
        let mean_mus: Vec<f64> = (0..b)
            .map(|k| {
                weights
                    .iter()
                    .zip(&grid)
                    .map(|(w, s)| w * sigma_mu_sq * stats.s1[k] / (s + stats.n[k] * sigma_mu_sq))
                    .sum::<f64>()
                    / total
            })
            .collect();
        (mean_sigma_sq, mean_mus)
    }

    fn batch_means_mcse(values: &[f64]) -> (f64, f64) {
        let batches = 200;
        let size = values.len() / batches;
        let means: Vec<f64> = (0..batches)
            .map(|k| values[k * size..(k + 1) * size].iter().sum::<f64>() / size as f64)
            .collect();
        let mean = means.iter().sum::<f64>() / batches as f64;
        let var =
            means.iter().map(|m| (m - mean) * (m - mean)).sum::<f64>() / (batches as f64 - 1.0);
        (mean, (var / batches as f64).sqrt())
    }

    #[test]
    fn fixed_tessellation_posterior_matches_quadrature() {
        let n = 12;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64 - 0.5).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| ((i * 7) % 13) as f64 / 26.0 - 0.25)
            .collect();
        let x = Data::new(xs, n, 1).unwrap();
        let lambda = 0.02;
        // With m' = 1 and a single-cell variance tessellation the
        // heteroscedastic model is the Gaussian model: its one cell value
        // is sigma^2 under the same inverse-gamma prior.
        let configs = [
            Config::new().with_m(1),
            Config::new()
                .with_m(1)
                .with_model(Model::Heteroscedastic)
                .with_m_var(1),
        ];
        for (config, (centres, seed)) in configs
            .iter()
            .flat_map(|c| [(vec![-0.35, 0.0, 0.3], 21_u64), (vec![0.0], 22)].map(|case| (c, case)))
        {
            let b = centres.len();
            let mut sampler = Sampler::pinned_prior(config, &x, &y, lambda, seed).unwrap();
            let fixed = Tessellation {
                centres,
                dims: vec![0],
                mus: vec![0.0; b],
            };
            sampler.fix_mean_tessellation(0, fixed);

            let stats = FixedCells::accumulate(sampler.mean_cells(0), &y, b);
            assert!(stats.n.iter().all(|&c| c > 0.0));
            let sigma_mu_sq = sampler.mean_sigma_mu_sq();
            let (ref_sigma_sq, ref_mus) =
                quadrature_reference(&stats, sampler.config.nu, lambda, sigma_mu_sq);

            for _ in 0..200 {
                sampler.conjugate_sweep();
            }
            let kept = 40_000;
            let mut sigma_sq = Vec::with_capacity(kept);
            let mut mus: Vec<Vec<f64>> = vec![Vec::with_capacity(kept); b];
            for _ in 0..kept {
                sampler.conjugate_sweep();
                sigma_sq.push(sampler.noise_variances()[0]);
                for (k, series) in mus.iter_mut().enumerate() {
                    series.push(sampler.tessellations()[0].mus[k]);
                }
            }
            let (mean, mcse) = batch_means_mcse(&sigma_sq);
            assert!(
                (mean - ref_sigma_sq).abs() < 4.0 * mcse,
                "sigma^2 {mean} vs {ref_sigma_sq} +- {mcse}"
            );
            for (k, series) in mus.iter().enumerate() {
                let (mean, mcse) = batch_means_mcse(series);
                assert!(
                    (mean - ref_mus[k]).abs() < 4.0 * mcse,
                    "mu_{k} {mean} vs {} +- {mcse}",
                    ref_mus[k]
                );
            }
        }
    }
}
