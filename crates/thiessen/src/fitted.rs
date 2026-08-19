//! The fitted model: the kept posterior draws, the scaling, and the
//! prediction surface.

use crate::config::Config;
use crate::data::{self, Data, Warning};
use crate::error::{Error, Result};
use crate::maths;
use crate::scaler::Scaler;
use crate::tessellation::Tessellation;

/// The kept posterior draws, scaled space: sigma^2 and the m tessellations
/// per draw.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "PosteriorParts")]
pub struct Posterior {
    sigma_sq: Vec<f64>,
    tessellations: Vec<Vec<Tessellation>>,
}

impl Posterior {
    pub(crate) fn empty() -> Self {
        Self {
            sigma_sq: Vec::new(),
            tessellations: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, sigma_sq: f64, tessellations: Vec<Tessellation>) {
        self.sigma_sq.push(sigma_sq);
        self.tessellations.push(tessellations);
    }

    /// Number of kept draws.
    pub fn n_draws(&self) -> usize {
        self.sigma_sq.len()
    }

    /// sigma^2 per draw, scaled space.
    pub fn sigma_sq(&self) -> &[f64] {
        &self.sigma_sq
    }

    /// The m tessellations of each draw.
    pub fn tessellations(&self) -> &[Vec<Tessellation>] {
        &self.tessellations
    }
}

#[derive(serde::Deserialize)]
struct PosteriorParts {
    sigma_sq: Vec<f64>,
    tessellations: Vec<Vec<Tessellation>>,
}

impl TryFrom<PosteriorParts> for Posterior {
    type Error = Error;

    fn try_from(parts: PosteriorParts) -> Result<Self> {
        let bad = |reason: &str| Error::InvalidSavedModel {
            reason: reason.into(),
        };
        if parts.sigma_sq.is_empty() || parts.sigma_sq.len() != parts.tessellations.len() {
            return Err(bad(
                "posterior needs at least one draw with one sigma^2 each",
            ));
        }
        if parts.sigma_sq.iter().any(|s| !(s.is_finite() && *s > 0.0)) {
            return Err(bad("sigma^2 draws must be finite and positive"));
        }
        let m = parts.tessellations[0].len();
        if m == 0 || parts.tessellations.iter().any(|d| d.len() != m) {
            return Err(bad(
                "every draw must hold the same positive number of tessellations",
            ));
        }
        Ok(Self {
            sigma_sq: parts.sigma_sq,
            tessellations: parts.tessellations,
        })
    }
}

/// A fitted model: the configuration, the scaling, the kept draws and the
/// fit-time warnings. Serialises through serde; loading validates the
/// payload.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "FittedParts")]
pub struct Fitted {
    config: Config,
    scaler: Scaler,
    posterior: Posterior,
    warnings: Vec<Warning>,
    in_sample_rmse: f64,
}

#[derive(serde::Deserialize)]
struct FittedParts {
    config: Config,
    scaler: Scaler,
    posterior: Posterior,
    warnings: Vec<Warning>,
    in_sample_rmse: f64,
}

impl TryFrom<FittedParts> for Fitted {
    type Error = Error;

    fn try_from(parts: FittedParts) -> Result<Self> {
        let bad = |reason: &str| Error::InvalidSavedModel {
            reason: reason.into(),
        };
        parts.config.validate()?;
        let p = parts.scaler.n_cols();
        for draw in parts.posterior.tessellations() {
            if draw.len() != parts.config.m {
                return Err(bad("draws do not hold m tessellations"));
            }
            if draw.iter().any(|t| t.dims().iter().any(|&d| d >= p)) {
                return Err(bad(
                    "a tessellation uses a covariate the scaler does not have",
                ));
            }
        }
        if !parts.in_sample_rmse.is_finite() {
            return Err(bad("in-sample RMSE must be finite"));
        }
        Ok(Self {
            config: parts.config,
            scaler: parts.scaler,
            posterior: parts.posterior,
            warnings: parts.warnings,
            in_sample_rmse: parts.in_sample_rmse,
        })
    }
}

/// A central credible interval for the mean function at one row.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// Lower end.
    pub lower: f64,
    /// Upper end.
    pub upper: f64,
}

impl Fitted {
    pub(crate) fn new(
        config: Config,
        scaler: Scaler,
        posterior: Posterior,
        warnings: Vec<Warning>,
        in_sample_rmse: f64,
    ) -> Self {
        Self {
            config,
            scaler,
            posterior,
            warnings,
            in_sample_rmse,
        }
    }

    /// Posterior mean of f(x) at each row of `x`, caller scale.
    ///
    /// # Errors
    ///
    /// `FeatureCountMismatch`, `NonFiniteFeature`.
    pub fn predict(&self, x: &Data) -> Result<Vec<f64>> {
        let per_draw = self.predict_draws(x)?;
        let n = x.n_rows();
        let n_draws = per_draw.len() as f64;
        let mut means = vec![0.0; n];
        for draw in &per_draw {
            for (mean, value) in means.iter_mut().zip(draw) {
                *mean += value;
            }
        }
        for mean in &mut means {
            *mean /= n_draws;
        }
        Ok(means)
    }

    /// f(x) at each row of `x` for every kept draw, draw-major
    /// (`n_draws` by `n_rows`), caller scale.
    ///
    /// # Errors
    ///
    /// `FeatureCountMismatch`, `NonFiniteFeature`.
    pub fn predict_draws(&self, x: &Data) -> Result<Vec<Vec<f64>>> {
        data::validate_predict(x, self.scaler.n_cols())?;
        let x_scaled = self.scaler.scale_x(x);
        let n = x.n_rows();
        Ok(self
            .posterior
            .tessellations()
            .iter()
            .map(|draw| {
                (0..n)
                    .map(|i| {
                        let row = x_scaled.row(i);
                        let sum: f64 = draw.iter().map(|t| t.value_at(row)).sum();
                        self.scaler.unscale_y(sum)
                    })
                    .collect()
            })
            .collect())
    }

    /// Posterior quantiles of f(x) at each row of `x` for each probability
    /// in `probs`, row-major (`n_rows` by `probs.len()`), caller scale; type
    /// 7 interpolation over the kept draws.
    ///
    /// # Errors
    ///
    /// `InvalidProbability` for a probability outside (0, 1) or an empty
    /// `probs`; the predict errors.
    pub fn predict_quantiles(&self, x: &Data, probs: &[f64]) -> Result<Vec<f64>> {
        if probs.is_empty() {
            return Err(Error::InvalidProbability { value: f64::NAN });
        }
        for &p in probs {
            check_probability(p)?;
        }
        let per_draw = self.predict_draws(x)?;
        let n = x.n_rows();
        let mut sorted = vec![0.0; per_draw.len()];
        let mut out = Vec::with_capacity(n * probs.len());
        for row in 0..n {
            for (slot, draw) in sorted.iter_mut().zip(&per_draw) {
                *slot = draw[row];
            }
            sorted.sort_by(f64::total_cmp);
            out.extend(probs.iter().map(|&p| maths::quantile_sorted(&sorted, p)));
        }
        Ok(out)
    }

    /// Central credible interval for f(x) at each row of `x` at `level`
    /// (the (1 - level) / 2 and (1 + level) / 2 posterior quantiles).
    ///
    /// # Errors
    ///
    /// `InvalidProbability` for `level` outside (0, 1); the predict errors.
    pub fn credible_interval(&self, x: &Data, level: f64) -> Result<Vec<Interval>> {
        check_probability(level)?;
        let tail = 0.5 * (1.0 - level);
        let q = self.predict_quantiles(x, &[tail, 1.0 - tail])?;
        Ok(q.chunks_exact(2)
            .map(|pair| Interval {
                lower: pair[0],
                upper: pair[1],
            })
            .collect())
    }

    /// Central posterior predictive interval for a new observation at each
    /// row of `x` at `level`: the quantiles of the equal-weight mixture over
    /// kept draws of N(f_d(x), sigma_d^2), found by bisection on the
    /// mixture CDF.
    ///
    /// # Errors
    ///
    /// `InvalidProbability` for `level` outside (0, 1); the predict errors.
    pub fn prediction_interval(&self, x: &Data, level: f64) -> Result<Vec<Interval>> {
        check_probability(level)?;
        let per_draw = self.predict_draws(x)?;
        let sigmas = self.sigma();
        let tail = 0.5 * (1.0 - level);
        let n = x.n_rows();
        let mut fits = vec![0.0; per_draw.len()];
        let mut out = Vec::with_capacity(n);
        for row in 0..n {
            for (fit, draw) in fits.iter_mut().zip(&per_draw) {
                *fit = draw[row];
            }
            out.push(Interval {
                lower: mixture_quantile(&fits, &sigmas, tail),
                upper: mixture_quantile(&fits, &sigmas, 1.0 - tail),
            });
        }
        Ok(out)
    }

    /// Pointwise log-likelihood ln N(y_i | f_d(x_i), sigma_d^2) per draw,
    /// draw-major (`n_draws` by `n_rows`).
    ///
    /// # Errors
    ///
    /// `RowCountMismatch`, `NonFiniteResponse`; the predict errors.
    pub fn log_likelihood(&self, x: &Data, y: &[f64]) -> Result<Vec<Vec<f64>>> {
        if y.len() != x.n_rows() {
            return Err(Error::RowCountMismatch {
                y_len: y.len(),
                x_rows: x.n_rows(),
            });
        }
        if let Some(row) = y.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteResponse { row });
        }
        let per_draw = self.predict_draws(x)?;
        let sigmas = self.sigma();
        let ln_2pi = maths::ln(2.0 * std::f64::consts::PI);
        Ok(per_draw
            .iter()
            .zip(&sigmas)
            .map(|(fits, &sigma)| {
                y.iter()
                    .zip(fits)
                    .map(|(&yi, &fit)| {
                        let z = (yi - fit) / sigma;
                        -0.5 * ln_2pi - maths::ln(sigma) - 0.5 * z * z
                    })
                    .collect()
            })
            .collect())
    }

    /// sigma per kept draw, caller scale: sqrt(sigma^2) times the training
    /// range of the response.
    pub fn sigma(&self) -> Vec<f64> {
        let range = self.scaler.y_range();
        self.posterior
            .sigma_sq()
            .iter()
            .map(|s| s.sqrt() * range)
            .collect()
    }

    /// Mean number of cells per tessellation, one value per kept draw.
    pub fn cell_counts(&self) -> Vec<f64> {
        self.posterior
            .tessellations()
            .iter()
            .map(|draw| draw.iter().map(|t| t.n_cells() as f64).sum::<f64>() / draw.len() as f64)
            .collect()
    }

    /// Mean number of active covariates per tessellation, one value per
    /// kept draw.
    pub fn dimension_counts(&self) -> Vec<f64> {
        self.posterior
            .tessellations()
            .iter()
            .map(|draw| draw.iter().map(|t| t.n_dims() as f64).sum::<f64>() / draw.len() as f64)
            .collect()
    }

    /// Share of active tessellation dimensions over all kept draws that
    /// fall on each covariate; sums to 1 (Chipman, George and McCulloch
    /// 2010, s. 5.1).
    pub fn variable_inclusion_proportions(&self) -> Vec<f64> {
        let mut counts = vec![0u64; self.scaler.n_cols()];
        let mut total = 0u64;
        for draw in self.posterior.tessellations() {
            for t in draw {
                for &dim in t.dims() {
                    counts[dim] += 1;
                    total += 1;
                }
            }
        }
        counts
            .iter()
            .map(|&c| c as f64 / total.max(1) as f64)
            .collect()
    }

    /// Number of kept draws.
    pub fn n_draws(&self) -> usize {
        self.posterior.n_draws()
    }

    /// The kept draws, scaled space.
    pub fn posterior(&self) -> &Posterior {
        &self.posterior
    }

    /// The scaling the model was fitted with.
    pub fn scaler(&self) -> &Scaler {
        &self.scaler
    }

    /// The configuration the model was fitted with, omega resolved.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Fit-time warnings.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// Root mean square of the posterior-mean fit against the training
    /// response, caller scale.
    pub fn in_sample_rmse(&self) -> f64 {
        self.in_sample_rmse
    }
}

fn check_probability(p: f64) -> Result<()> {
    if p.is_finite() && p > 0.0 && p < 1.0 {
        Ok(())
    } else {
        Err(Error::InvalidProbability { value: p })
    }
}

/// CDF at `t` of the equal-weight mixture of N(fit_d, sigma_d^2).
fn mixture_cdf(fits: &[f64], sigmas: &[f64], t: f64) -> f64 {
    let sum: f64 = fits
        .iter()
        .zip(sigmas)
        .map(|(&fit, &sigma)| maths::normal_cdf((t - fit) / sigma))
        .sum();
    sum / fits.len() as f64
}

/// Quantile `p` of the mixture by bisection on [`mixture_cdf`] over a
/// bracket of the fits padded by 39 sigma_max (where Phi is 0 or 1 to
/// double precision).
fn mixture_quantile(fits: &[f64], sigmas: &[f64], p: f64) -> f64 {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut sigma_max = 0.0_f64;
    for (&fit, &sigma) in fits.iter().zip(sigmas) {
        lo = lo.min(fit);
        hi = hi.max(fit);
        sigma_max = sigma_max.max(sigma);
    }
    let pad = 39.0 * sigma_max;
    let (mut lo, mut hi) = (lo - pad, hi + pad);
    for _ in 0..128 {
        let mid = 0.5 * (lo + hi);
        if mixture_cdf(fits, sigmas, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit;

    fn fitted() -> (Fitted, Data, Vec<f64>) {
        let n = 30;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
        let x = Data::new(xs, n, 1).unwrap();
        let config = Config::new().with_m(10).with_burn_in(20).with_draws(30);
        (fit(&config, &x, &y, 42).unwrap(), x, y)
    }

    #[test]
    fn prediction_surface_shapes_and_order() {
        let (model, x, y) = fitted();
        let mean = model.predict(&x).unwrap();
        let draws = model.predict_draws(&x).unwrap();
        assert_eq!(mean.len(), 30);
        assert_eq!((draws.len(), draws[0].len()), (30, 30));
        let q = model.predict_quantiles(&x, &[0.1, 0.5, 0.9]).unwrap();
        assert_eq!(q.len(), 90);
        for row in 0..30 {
            assert!(q[row * 3] <= q[row * 3 + 1] && q[row * 3 + 1] <= q[row * 3 + 2]);
        }
        let ci = model.credible_interval(&x, 0.9).unwrap();
        let pi = model.prediction_interval(&x, 0.9).unwrap();
        for (c, p) in ci.iter().zip(&pi) {
            assert!(p.lower <= c.lower && c.upper <= p.upper);
        }
        let ll = model.log_likelihood(&x, &y).unwrap();
        assert_eq!((ll.len(), ll[0].len()), (30, 30));
        assert!(ll.iter().flatten().all(|v| v.is_finite()));
        assert_eq!(model.sigma().len(), 30);
        assert_eq!(model.cell_counts().len(), 30);
        assert_eq!(model.dimension_counts(), vec![1.0; 30]);
        assert_eq!(model.variable_inclusion_proportions(), vec![1.0]);
        assert!(model.in_sample_rmse() < 0.5);
        assert!(model.warnings().is_empty());
    }

    #[test]
    fn predict_errors() {
        let (model, _, y) = fitted();
        let wrong = Data::new(vec![0.0, 1.0], 1, 2).unwrap();
        assert!(matches!(
            model.predict(&wrong),
            Err(Error::FeatureCountMismatch {
                expected: 1,
                found: 2
            })
        ));
        let x = Data::new(vec![0.5], 1, 1).unwrap();
        assert!(matches!(
            model.predict_quantiles(&x, &[1.5]),
            Err(Error::InvalidProbability { .. })
        ));
        assert!(model.predict_quantiles(&x, &[]).is_err());
        assert!(model.credible_interval(&x, 0.0).is_err());
        assert!(matches!(
            model.log_likelihood(&x, &y),
            Err(Error::RowCountMismatch { .. })
        ));
        let empty = Data::new(vec![], 0, 1).unwrap();
        assert!(model.predict(&empty).unwrap().is_empty());
    }

    #[test]
    fn serde_round_trip_preserves_predictions() {
        let (model, x, _) = fitted();
        let json = serde_json::to_string(&model).unwrap();
        let loaded: Fitted = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded, model);
        assert_eq!(loaded.predict(&x).unwrap(), model.predict(&x).unwrap());
        let corrupt = json.replace("\"in_sample_rmse\":", "\"in_sample_rmse\":1e999,\"x\":");
        assert!(serde_json::from_str::<Fitted>(&corrupt).is_err());
    }

    #[test]
    fn mixture_quantile_of_one_normal() {
        let q = mixture_quantile(&[1.0], &[2.0], 0.975);
        assert!((q - (1.0 + 2.0 * 1.959_963_984_540_054)).abs() < 1e-9);
    }
}
