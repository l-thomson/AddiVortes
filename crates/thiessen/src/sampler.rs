//! The Gibbs sampler with Metropolis-Hastings structural moves (Stone and
//! Gosling 2025, Algorithm 1): one sweep per `step`, the caller's loop
//! deciding burn-in and thinning through `keep`, and `finish` producing the
//! fitted model.

use crate::cells::CellStats;
use crate::config::Config;
use crate::data::{self, Data, Warning};
use crate::error::{Error, Result};
use crate::fitted::{Fitted, Posterior};
use crate::maths;
use crate::moves::{self, Prior};
use crate::rng::{self, Rng};
use crate::scaler::{self, Scaler};
use crate::tessellation::{Assignment, Tessellation};

/// The sampler state for one chain. Construct with [`Sampler::new`], advance
/// with [`step`](Sampler::step), record draws with [`keep`](Sampler::keep),
/// and close with [`finish`](Sampler::finish).
///
/// The response is on the caller's scale through the affine map frozen at
/// construction; [`set_response`](Sampler::set_response) and
/// [`fitted_values`](Sampler::fitted_values) speak that scale. The sampler
/// owns its RNG, seeded at construction; a caller's loop consumes none of it.
#[derive(Debug, Clone)]
pub struct Sampler {
    rng: Rng,
    config: Config,
    scaler: Scaler,
    warnings: Vec<Warning>,
    /// Scaled design.
    x: Data,
    /// Scaled working response.
    y: Vec<f64>,
    tessellations: Vec<Tessellation>,
    assignments: Vec<Assignment>,
    /// Running ensemble fit, scaled space.
    fit: Vec<f64>,
    sigma_sq: f64,
    /// Per-observation precision, 1 / sigma_sq for every observation.
    precision: Vec<f64>,
    prior: Prior,
    sigma_mu_sq: f64,
    /// Calibrated scale of the prior sigma^2 ~ nu lambda / chi^2_nu.
    lambda: f64,
    kept: Posterior,
}

impl Sampler {
    /// A sampler over raw data, its RNG seeded from `seed`.
    ///
    /// # Errors
    ///
    /// [`Config::validate`], then the data checks: row counts, at least two
    /// observations, finite values, non-constant response and columns, and
    /// omega <= p.
    pub fn new(config: &Config, x: &Data, y: &[f64], seed: u64) -> Result<Self> {
        config.validate()?;
        let p = x.n_cols();
        let omega = config.omega_for(p);
        data::validate_fit(x, y, omega)?;
        let mut config = config.clone();
        config.omega = Some(omega);
        let warnings = data::fit_warnings(x);
        let (scaler, x_scaled, y_scaled) = Scaler::fit(x, y);
        let n = x_scaled.n_rows();

        let sigma_hat = scaler::sigma_hat(&x_scaled, &y_scaled);
        let lambda = scaler::calibrate_lambda(config.nu, config.q, sigma_hat);
        let sigma_mu_sq = scaler::sigma_mu_sq(config.k, config.m);
        let prior = Prior {
            p,
            omega,
            lambda_c: config.lambda_c,
            sigma_c: config.sigma_c,
        };

        // Initial state: m single-cell tessellations on one covariate each,
        // every cell mean ybar / m so the ensemble fit starts at ybar.
        let mut rng = rng::chain_rng(seed);
        let mean_y = y_scaled.iter().sum::<f64>() / n as f64;
        let tessellations: Vec<Tessellation> = (0..config.m)
            .map(|_| {
                let dim = rng::uniform_index(p, &mut rng);
                let centre = config.sigma_c * rng::standard_normal(&mut rng);
                Tessellation {
                    centres: vec![centre],
                    dims: vec![dim],
                    mus: vec![mean_y / config.m as f64],
                }
            })
            .collect();
        let assignments = tessellations
            .iter()
            .map(|t| Assignment::full(&x_scaled, t))
            .collect();
        let sigma_sq = sigma_hat * sigma_hat;

        Ok(Self {
            rng,
            config,
            scaler,
            warnings,
            x: x_scaled,
            y: y_scaled,
            tessellations,
            assignments,
            fit: vec![mean_y; n],
            sigma_sq,
            precision: vec![1.0 / sigma_sq; n],
            prior,
            sigma_mu_sq,
            lambda,
            kept: Posterior::empty(),
        })
    }

    /// One sweep: sigma^2 | rest, then for each tessellation in turn a
    /// structural move, accept or reject, and the cell means | rest.
    pub fn step(&mut self) {
        self.draw_sigma_sq();
        for j in 0..self.tessellations.len() {
            self.backfit(j);
        }
    }

    /// sigma^2 | y, F ~ Inv-Gamma((nu + n) / 2, (nu lambda + sum r_i^2) / 2)
    /// with r = y - F.
    fn draw_sigma_sq(&mut self) {
        let n = self.y.len();
        let rss: f64 = self
            .y
            .iter()
            .zip(&self.fit)
            .map(|(y, f)| (y - f) * (y - f))
            .sum();
        let shape = 0.5 * (self.config.nu + n as f64);
        let scale = 2.0 / (self.config.nu * self.lambda + rss);
        self.sigma_sq = 1.0 / rng::gamma(shape, scale, &mut self.rng);
        let precision = 1.0 / self.sigma_sq;
        self.precision.iter_mut().for_each(|w| *w = precision);
    }

    /// The backfitting update of tessellation `j`: partial residuals, one
    /// structural move with the empty-cell guard, the cell means, and the
    /// running fit.
    fn backfit(&mut self, j: usize) {
        let n = self.y.len();
        let current = &self.tessellations[j];
        let cells = &self.assignments[j].cells;
        let residuals: Vec<f64> = (0..n)
            .map(|i| self.y[i] - self.fit[i] + current.mus[cells[i]])
            .collect();

        let m = moves::select(current, &self.prior, &mut self.rng);
        let proposal = moves::propose(m, current, &self.prior, &mut self.rng);
        let proposed_assignment =
            self.assignments[j].updated(&self.x, &proposal.tessellation, proposal.delta);
        let proposed_stats = CellStats::accumulate(
            &proposed_assignment.cells,
            &residuals,
            &self.precision,
            proposal.tessellation.n_cells(),
        );
        // A proposal leaving a cell empty is rejected before the acceptance
        // draw, so no uniform is consumed.
        let mut stats = None;
        if proposed_stats.all_occupied() {
            let current_stats =
                CellStats::accumulate(cells, &residuals, &self.precision, current.n_cells());
            let log_alpha = proposed_stats.log_marginal(self.sigma_mu_sq)
                - current_stats.log_marginal(self.sigma_mu_sq)
                + proposal.log_structure_ratio
                + moves::log_selection_ratio(m, current, &proposal.tessellation, &self.prior);
            debug_assert!(!log_alpha.is_nan());
            let u = rng::uniform(&mut self.rng);
            if maths::ln(u) < log_alpha {
                self.tessellations[j] = proposal.tessellation;
                self.assignments[j] = proposed_assignment;
                stats = Some(proposed_stats);
            } else {
                stats = Some(current_stats);
            }
        }
        let tessellation = &mut self.tessellations[j];
        let cells = &self.assignments[j].cells;
        let stats = stats.unwrap_or_else(|| {
            CellStats::accumulate(cells, &residuals, &self.precision, tessellation.n_cells())
        });
        tessellation.mus = stats.draw_means(self.sigma_mu_sq, &mut self.rng);
        for i in 0..n {
            self.fit[i] = self.y[i] - residuals[i] + tessellation.mus[cells[i]];
        }
    }

    /// Record the current state as a posterior draw.
    pub fn keep(&mut self) {
        self.kept.push(self.sigma_sq, self.tessellations.clone());
    }

    /// Number of draws kept so far.
    pub fn n_kept(&self) -> usize {
        self.kept.n_draws()
    }

    /// Replace the response (caller scale), keeping the tessellations, the
    /// cell means and sigma^2. The partial residuals of the next sweep use
    /// the new response.
    ///
    /// # Errors
    ///
    /// `RowCountMismatch` or `NonFiniteResponse`.
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
        for (slot, &v) in self.y.iter_mut().zip(y) {
            *slot = self.scaler.scale_y(v);
        }
        Ok(())
    }

    /// The current ensemble fit at the training rows, caller scale.
    pub fn fitted_values(&self) -> Vec<f64> {
        self.fit.iter().map(|&f| self.scaler.unscale_y(f)).collect()
    }

    /// The current sigma^2, scaled space.
    pub fn sigma_sq(&self) -> f64 {
        self.sigma_sq
    }

    /// The current tessellations, scaled space.
    pub fn tessellations(&self) -> &[Tessellation] {
        &self.tessellations
    }

    /// The scaling frozen at construction.
    pub fn scaler(&self) -> &Scaler {
        &self.scaler
    }

    /// The configuration, with omega resolved.
    pub fn config(&self) -> &Config {
        &self.config
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
        let mut mean_fit = vec![0.0; n];
        for draw in self.kept.tessellations() {
            for (i, slot) in mean_fit.iter_mut().enumerate() {
                let row = self.x.row(i);
                *slot += draw.iter().map(|t| t.value_at(row)).sum::<f64>();
            }
        }
        let n_draws = self.kept.n_draws() as f64;
        let range = self.scaler.y_range();
        let in_sample_rmse = (mean_fit
            .iter()
            .zip(&self.y)
            .map(|(f, y)| {
                let r = (f / n_draws - y) * range;
                r * r
            })
            .sum::<f64>()
            / n as f64)
            .sqrt();
        Ok(Fitted::new(
            self.config,
            self.scaler,
            self.kept,
            self.warnings,
            in_sample_rmse,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit;

    fn toy(n: usize) -> (Data, Vec<f64>) {
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
        let x = Data::new(xs, n, 1).unwrap();
        (x, y)
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
            for i in 0..40 {
                let row = sampler.x.row(i);
                let sum: f64 = sampler.tessellations.iter().map(|t| t.value_at(row)).sum();
                assert!((sum - sampler.fit[i]).abs() < 1e-9);
            }
            for (t, a) in sampler.tessellations.iter().zip(&sampler.assignments) {
                assert_eq!(*a, Assignment::full(&sampler.x, t));
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
}
