//! The fitted model: the kept posterior draws, the scaling, and the
//! prediction surface, with the semantics of each method per outcome
//! model.

use crate::config::Config;
use crate::config::Outcome;
use crate::data::{self, Data, Warning};
use crate::error::{Error, Result};
use crate::geometry::Geometry;
use crate::maths;
use crate::scaler::Scaler;
use crate::tessellation::Tessellation;

/// The kept posterior draws, scaled space: the m mean tessellations per
/// draw; sigma^2 per draw under the Gaussian model; the m' variance
/// tessellations per draw under the heteroscedastic model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "PosteriorParts")]
pub struct Posterior {
    sigma_sq: Vec<f64>,
    tessellations: Vec<Vec<Tessellation>>,
    variance_tessellations: Vec<Vec<Tessellation>>,
}

impl Posterior {
    pub(crate) fn empty() -> Self {
        Self {
            sigma_sq: Vec::new(),
            tessellations: Vec::new(),
            variance_tessellations: Vec::new(),
        }
    }

    pub(crate) fn push(
        &mut self,
        sigma_sq: Option<f64>,
        tessellations: Vec<Tessellation>,
        variance_tessellations: Option<Vec<Tessellation>>,
    ) {
        self.sigma_sq.extend(sigma_sq);
        self.tessellations.push(tessellations);
        self.variance_tessellations.extend(variance_tessellations);
    }

    /// Number of kept draws.
    pub fn n_draws(&self) -> usize {
        self.tessellations.len()
    }

    /// sigma^2 per draw, scaled space; empty outside the Gaussian model.
    pub fn sigma_sq(&self) -> &[f64] {
        &self.sigma_sq
    }

    /// The m mean tessellations of each draw.
    pub fn tessellations(&self) -> &[Vec<Tessellation>] {
        &self.tessellations
    }

    /// The m' variance tessellations of each draw; empty outside the
    /// heteroscedastic model.
    pub fn variance_tessellations(&self) -> &[Vec<Tessellation>] {
        &self.variance_tessellations
    }

    pub(crate) fn extend(&mut self, other: &Self) {
        self.sigma_sq.extend_from_slice(&other.sigma_sq);
        self.tessellations.extend_from_slice(&other.tessellations);
        self.variance_tessellations
            .extend_from_slice(&other.variance_tessellations);
    }
}

#[derive(serde::Deserialize)]
struct PosteriorParts {
    sigma_sq: Vec<f64>,
    tessellations: Vec<Vec<Tessellation>>,
    #[serde(default)]
    variance_tessellations: Vec<Vec<Tessellation>>,
}

impl TryFrom<PosteriorParts> for Posterior {
    type Error = Error;

    fn try_from(parts: PosteriorParts) -> Result<Self> {
        let bad = |reason: &str| Error::InvalidSavedModel {
            reason: reason.into(),
        };
        let n_draws = parts.tessellations.len();
        if n_draws == 0 {
            return Err(bad("posterior needs at least one draw"));
        }
        if !(parts.sigma_sq.is_empty() || parts.sigma_sq.len() == n_draws) {
            return Err(bad("sigma^2 draws must be absent or one per draw"));
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
        if !(parts.variance_tessellations.is_empty()
            || parts.variance_tessellations.len() == n_draws)
        {
            return Err(bad(
                "variance tessellations must be absent or one set per draw",
            ));
        }
        if let Some(first) = parts.variance_tessellations.first() {
            let m_var = first.len();
            if m_var == 0
                || parts
                    .variance_tessellations
                    .iter()
                    .any(|d| d.len() != m_var)
            {
                return Err(bad(
                    "every draw must hold the same positive number of variance tessellations",
                ));
            }
            if parts
                .variance_tessellations
                .iter()
                .flatten()
                .any(|t| t.mus().iter().any(|v| *v <= 0.0))
            {
                return Err(bad("variance cell values must be positive"));
            }
        }
        Ok(Self {
            sigma_sq: parts.sigma_sq,
            tessellations: parts.tessellations,
            variance_tessellations: parts.variance_tessellations,
        })
    }
}

/// A fitted model: the configuration, the scaling, the kept draws and the
/// fit-time warnings. Serialises through serde; loading validates the
/// payload against the model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "FittedParts")]
pub struct Fitted {
    config: Config,
    scaler: Scaler,
    posterior: Posterior,
    warnings: Vec<Warning>,
    in_sample_rmse: f64,
    /// Levels of each categorical column; empty for the other columns.
    categories: Vec<Vec<f64>>,
}

#[derive(serde::Deserialize)]
struct FittedParts {
    config: Config,
    scaler: Scaler,
    posterior: Posterior,
    warnings: Vec<Warning>,
    in_sample_rmse: f64,
    #[serde(default)]
    categories: Vec<Vec<f64>>,
}

impl TryFrom<FittedParts> for Fitted {
    type Error = Error;

    fn try_from(parts: FittedParts) -> Result<Self> {
        let bad = |reason: &str| Error::InvalidSavedModel {
            reason: reason.into(),
        };
        parts.config.validate()?;
        let p = parts.scaler.n_cols();
        // A save from before categorical levels were stored carries none.
        let categories = if parts.categories.is_empty() {
            vec![Vec::new(); p]
        } else {
            parts.categories
        };
        let geometry =
            Geometry::with_categories(&parts.config.mean_params.geometry.metric, p, &categories)?;
        #[cfg(feature = "experimental")]
        geometry.with_precision(parts.config.mean_params.geometry.precision.as_deref())?;
        #[cfg(not(feature = "experimental"))]
        drop(geometry);
        let uses_covariates = |draws: &[Vec<Tessellation>]| {
            draws
                .iter()
                .flatten()
                .all(|t| t.dims().iter().all(|&d| d < p))
        };
        for draw in parts.posterior.tessellations() {
            if draw.len() != parts.config.mean_tessellations() {
                return Err(bad("draws do not hold m tessellations"));
            }
        }
        if !uses_covariates(parts.posterior.tessellations())
            || !uses_covariates(parts.posterior.variance_tessellations())
        {
            return Err(bad(
                "a tessellation uses a covariate the scaler does not have",
            ));
        }
        let n_draws = parts.posterior.n_draws();
        let has_ensemble = parts.config.variance_tessellations() > 0;
        let has_global_sigma_sq =
            parts.config.outcome.sigma2_mode().samples_global_sigma_sq() && !has_ensemble;
        if (parts.posterior.sigma_sq().len() == n_draws) != has_global_sigma_sq {
            return Err(bad(
                "sigma^2 draws are present exactly where the scale is sampled globally",
            ));
        }
        if (parts.posterior.variance_tessellations().len() == n_draws) != has_ensemble {
            return Err(bad(
                "variance tessellations are present exactly under a variance ensemble",
            ));
        }
        if has_ensemble
            && parts.posterior.variance_tessellations()[0].len()
                != parts.config.variance_tessellations()
        {
            return Err(bad(
                "draws do not hold the variance-ensemble tessellation count",
            ));
        }
        if matches!(parts.config.outcome, Outcome::Probit(_)) {
            match parts.config.offset() {
                Some(c) if c.is_finite() => {}
                _ => return Err(bad("a probit fit carries a finite offset")),
            }
            if parts.scaler.y_range() != 1.0 || parts.scaler.y_min() != -0.5 {
                return Err(bad("a probit fit leaves the response unscaled"));
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
            categories,
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
        categories: Vec<Vec<f64>>,
    ) -> Self {
        Self {
            config,
            scaler,
            posterior,
            warnings,
            in_sample_rmse,
            categories,
        }
    }

    /// The fitted model's name: "gaussian", "probit", or "heteroscedastic"
    /// for the Gaussian outcome with its variance ensemble attached.
    pub fn model_name(&self) -> &'static str {
        self.config.model_name()
    }

    /// The observation model the draws were fitted under.
    pub fn outcome(&self) -> &Outcome {
        &self.config.outcome
    }

    /// Whether the spread is the variance ensemble's product s^2(x).
    pub fn has_variance_ensemble(&self) -> bool {
        self.config.variance_tessellations() > 0
    }

    /// The kept draws of `fits`, in chain order, as one fitted model.
    ///
    /// # Arguments
    ///
    /// `fits` are chains of the same model fitted to `x` and `y`, keyed by
    /// [`chain_seed`](crate::chain_seed). The in-sample RMSE is that of the
    /// pooled posterior mean.
    ///
    /// # Errors
    ///
    /// `MismatchedChains` for an empty `fits` or chains whose
    /// configuration, scaling or category levels differ;
    /// `RowCountMismatch`; the [`predict`](Self::predict) errors.
    pub fn pool(fits: &[Self], x: &Data, y: &[f64]) -> Result<Self> {
        let mismatch = |reason: &str| Error::MismatchedChains {
            reason: reason.into(),
        };
        let (first, rest) = fits.split_first().ok_or_else(|| mismatch("no chains"))?;
        for fit in rest {
            if fit.config != first.config {
                return Err(mismatch("configurations differ"));
            }
            if fit.scaler != first.scaler || fit.categories != first.categories {
                return Err(mismatch("chains were fitted to different data"));
            }
        }
        if y.len() != x.n_rows() {
            return Err(Error::RowCountMismatch {
                y_len: y.len(),
                x_rows: x.n_rows(),
            });
        }
        let mut posterior = Posterior::empty();
        let mut warnings: Vec<Warning> = Vec::new();
        for fit in fits {
            posterior.extend(&fit.posterior);
            for warning in &fit.warnings {
                if !warnings.contains(warning) {
                    warnings.push(*warning);
                }
            }
        }
        let mut pooled = Self::new(
            first.config.clone(),
            first.scaler.clone(),
            posterior,
            warnings,
            0.0,
            first.categories.clone(),
        );
        let mean = pooled.predict(x)?;
        let n = y.len() as f64;
        pooled.in_sample_rmse = (mean
            .iter()
            .zip(y)
            .map(|(f, y)| (f - y) * (f - y))
            .sum::<f64>()
            / n)
            .sqrt();
        Ok(pooled)
    }

    fn not_applicable(&self, method: &str) -> Error {
        Error::NotApplicable {
            method: method.into(),
            model: self.config.model_name().into(),
        }
    }

    /// Posterior mean at each row of `x`, caller scale: of f(x), or of
    /// P(y = 1 | x) = Phi(c + f(x)) under the probit model. Under the
    /// tobit model the quantity is the uncensored f(x), the latent mean;
    /// under the AFT model it is f(x) on the log-time scale (the BART
    /// package's `yhat` convention for `abart`).
    ///
    /// # Errors
    ///
    /// `FeatureCountMismatch`, `NonFiniteFeature`, `InvalidCategoryCode`.
    pub fn predict(&self, x: &Data) -> Result<Vec<f64>> {
        Ok(column_means(&self.predict_draws(x)?, x.n_rows()))
    }

    /// The quantity of [`predict`](Self::predict) at each row of `x` for
    /// every kept draw, draw-major (`n_draws` by `n_rows`), caller scale.
    ///
    /// # Errors
    ///
    /// `FeatureCountMismatch`, `NonFiniteFeature`, `InvalidCategoryCode`.
    pub fn predict_draws(&self, x: &Data) -> Result<Vec<Vec<f64>>> {
        let mut latent = self.predict_latent(x)?;
        if matches!(self.config.outcome, Outcome::Probit(_)) {
            for draw in &mut latent {
                for v in draw {
                    *v = maths::normal_cdf(*v);
                }
            }
        }
        Ok(latent)
    }

    /// The mean function at each row of `x` for every kept draw, draw-major
    /// (`n_draws` by `n_rows`), caller scale: f(x), or the latent mean
    /// c + f(x) under the probit model.
    ///
    /// # Errors
    ///
    /// `FeatureCountMismatch`, `NonFiniteFeature`, `InvalidCategoryCode`.
    pub fn predict_latent(&self, x: &Data) -> Result<Vec<Vec<f64>>> {
        data::validate_predict(x, self.scaler.n_cols())?;
        let geometry = self.geometry()?;
        geometry.check_codes(x)?;
        let x_scaled = self.scaler.scale_x(x);
        let n = x.n_rows();
        let offset = self.offset();
        Ok(self
            .posterior
            .tessellations()
            .iter()
            .map(|draw| {
                (0..n)
                    .map(|i| {
                        let row = x_scaled.row(i);
                        let sum: f64 = draw.iter().map(|t| t.value_at(row, &geometry)).sum();
                        self.scaler.unscale_y(sum) + offset
                    })
                    .collect()
            })
            .collect())
    }

    /// The column structure of the fit, from the configuration and the
    /// stored categorical levels.
    fn geometry(&self) -> Result<Geometry> {
        let geometry = Geometry::with_categories(
            &self.config.mean_params.geometry.metric,
            self.scaler.n_cols(),
            &self.categories,
        )?;
        #[cfg(feature = "experimental")]
        let geometry =
            geometry.with_precision(self.config.mean_params.geometry.precision.as_deref())?;
        Ok(geometry)
    }

    /// The variance of y given f at each row of `x` for every kept draw,
    /// draw-major (`n_draws` by `n_rows`), caller scale: sigma_d^2 under
    /// the Gaussian model (constant across rows); s_d^2(x), the product of
    /// the variance tessellations, under the heteroscedastic model (the
    /// square of `rbart`'s `sdraws`).
    ///
    /// # Errors
    ///
    /// `NotApplicable` under the probit model; `FeatureCountMismatch`,
    /// `NonFiniteFeature`, `InvalidCategoryCode`.
    pub fn predict_variance(&self, x: &Data) -> Result<Vec<Vec<f64>>> {
        data::validate_predict(x, self.scaler.n_cols())?;
        let n = x.n_rows();
        let range_sq = self.scaler.y_range() * self.scaler.y_range();
        if matches!(self.config.outcome, Outcome::Probit(_)) {
            return Err(self.not_applicable("predict_variance"));
        }
        if self.has_variance_ensemble() {
            let geometry = self.geometry()?;
            geometry.check_codes(x)?;
            let x_scaled = self.scaler.scale_x(x);
            return Ok(self
                .posterior
                .variance_tessellations()
                .iter()
                .map(|draw| {
                    (0..n)
                        .map(|i| {
                            let row = x_scaled.row(i);
                            draw.iter()
                                .map(|t| t.value_at(row, &geometry))
                                .product::<f64>()
                                * range_sq
                        })
                        .collect()
                })
                .collect());
        }
        Ok(self
            .posterior
            .sigma_sq()
            .iter()
            .map(|s| vec![s * range_sq; n])
            .collect())
    }

    /// Posterior quantiles of the quantity of [`predict`](Self::predict) at
    /// each row of `x` for each probability in `probs`, row-major
    /// (`n_rows` by `probs.len()`), caller scale; type 7 interpolation over
    /// the kept draws.
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

    /// Central credible interval for the quantity of
    /// [`predict`](Self::predict) at each row of `x` at `level` (the
    /// (1 - level) / 2 and (1 + level) / 2 posterior quantiles); on the
    /// probability scale under the probit model.
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
    /// kept draws of N(f_d(x), s_d^2(x)), found by bisection on the mixture
    /// CDF. Under the tobit model the predictive is censored, so the ends
    /// are clamped to the limits (censoring is monotone, which makes the
    /// clamp the exact quantile).
    ///
    /// # Errors
    ///
    /// `NotApplicable` under the probit model, which has no continuous
    /// predictive distribution; `InvalidProbability` for `level` outside
    /// (0, 1); the predict errors.
    pub fn prediction_interval(&self, x: &Data, level: f64) -> Result<Vec<Interval>> {
        check_probability(level)?;
        if matches!(self.config.outcome, Outcome::Probit(_)) {
            return Err(self.not_applicable("prediction_interval"));
        }
        let per_draw = self.predict_draws(x)?;
        let variances = self.predict_variance(x)?;
        let tail = 0.5 * (1.0 - level);
        let n = x.n_rows();
        let mut fits = vec![0.0; per_draw.len()];
        let mut sigmas = vec![0.0; per_draw.len()];
        let mut out = Vec::with_capacity(n);
        for row in 0..n {
            for ((fit, sigma), (draw, variance)) in fits
                .iter_mut()
                .zip(&mut sigmas)
                .zip(per_draw.iter().zip(&variances))
            {
                *fit = draw[row];
                *sigma = variance[row].sqrt();
            }
            out.push(Interval {
                lower: mixture_quantile(&fits, &sigmas, tail),
                upper: mixture_quantile(&fits, &sigmas, 1.0 - tail),
            });
        }
        #[cfg(feature = "experimental")]
        if let Outcome::Tobit(params) = &self.config.outcome {
            let lo = params.lower.unwrap_or(f64::NEG_INFINITY);
            let hi = params.upper.unwrap_or(f64::INFINITY);
            for interval in &mut out {
                interval.lower = interval.lower.clamp(lo, hi);
                interval.upper = interval.upper.clamp(lo, hi);
            }
        }
        Ok(out)
    }

    /// Pointwise log-likelihood per draw, draw-major (`n_draws` by
    /// `n_rows`): ln N(y_i | f_d(x_i), s_d^2(x_i)), or under the probit
    /// model y_i ln p_d(x_i) + (1 - y_i) ln(1 - p_d(x_i)) with p_d the
    /// draw's P(y = 1 | x). Under the tobit model a row at a limit takes
    /// its censored term, ln Phi((lower - f_d) / s_d) or
    /// ln Phi((f_d - upper) / s_d), and the Normal log density otherwise.
    /// `NotApplicable` under the AFT model, whose pointwise likelihood
    /// needs the event indicator:
    /// [`log_likelihood_survival`](Self::log_likelihood_survival).
    ///
    /// # Errors
    ///
    /// `RowCountMismatch`, `NonFiniteResponse`, `InvalidLabel` under the
    /// probit model, `ResponseBeyondLimit` under the tobit model; the
    /// predict errors.
    pub fn log_likelihood(&self, x: &Data, y: &[f64]) -> Result<Vec<Vec<f64>>> {
        #[cfg(feature = "experimental")]
        if matches!(self.config.outcome, Outcome::Aft(_)) {
            return Err(self.not_applicable("log_likelihood"));
        }
        if y.len() != x.n_rows() {
            return Err(Error::RowCountMismatch {
                y_len: y.len(),
                x_rows: x.n_rows(),
            });
        }
        if let Some(row) = y.iter().position(|v| !v.is_finite()) {
            return Err(Error::NonFiniteResponse { row });
        }
        if matches!(self.config.outcome, Outcome::Probit(_)) {
            if let Some(row) = y.iter().position(|&v| v != 0.0 && v != 1.0) {
                return Err(Error::InvalidLabel { row });
            }
            let per_draw = self.predict_draws(x)?;
            return Ok(per_draw
                .iter()
                .map(|probs| {
                    y.iter()
                        .zip(probs)
                        .map(|(&yi, &p)| {
                            if yi == 1.0 {
                                maths::ln(p)
                            } else {
                                maths::ln(1.0 - p)
                            }
                        })
                        .collect()
                })
                .collect());
        }
        let per_draw = self.predict_draws(x)?;
        let variances = self.predict_variance(x)?;
        let ln_2pi = maths::ln(2.0 * std::f64::consts::PI);
        #[cfg(feature = "experimental")]
        if let Outcome::Tobit(params) = &self.config.outcome {
            let (lower, upper) = (params.lower, params.upper);
            let beyond = |v: f64| {
                lower.is_some_and(|limit| v < limit) || upper.is_some_and(|limit| v > limit)
            };
            if let Some(row) = y.iter().position(|&v| beyond(v)) {
                return Err(Error::ResponseBeyondLimit { row });
            }
            return Ok(per_draw
                .iter()
                .zip(&variances)
                .map(|(fits, variance)| {
                    y.iter()
                        .zip(fits.iter().zip(variance))
                        .map(|(&yi, (&fit, &var))| {
                            let sd = var.sqrt();
                            if lower == Some(yi) {
                                maths::ln(maths::normal_cdf((yi - fit) / sd))
                            } else if upper == Some(yi) {
                                maths::ln(maths::normal_cdf((fit - yi) / sd))
                            } else {
                                let z = (yi - fit) * (yi - fit) / var;
                                -0.5 * (ln_2pi + maths::ln(var) + z)
                            }
                        })
                        .collect()
                })
                .collect());
        }
        Ok(per_draw
            .iter()
            .zip(&variances)
            .map(|(fits, variance)| {
                y.iter()
                    .zip(fits.iter().zip(variance))
                    .map(|(&yi, (&fit, &var))| {
                        let z = (yi - fit) * (yi - fit) / var;
                        -0.5 * (ln_2pi + maths::ln(var) + z)
                    })
                    .collect()
            })
            .collect())
    }

    /// Pointwise survival log-likelihood per draw under the AFT model,
    /// draw-major (`n_draws` by `n_rows`): ln N(ln t_i; f_d(x_i),
    /// s_d^2(x_i)) at an event and ln Phi((f_d(x_i) - ln t_i) / s_d(x_i))
    /// at a censored row, the log-time density and the survival
    /// probability of the lognormal AFT model. Experimental
    /// (`docs/experimental.md`).
    ///
    /// # Errors
    ///
    /// `NotApplicable` under another model; `RowCountMismatch`,
    /// `EventCountMismatch`, `InvalidSurvivalTime`; the predict errors.
    #[cfg(feature = "experimental")]
    #[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
    pub fn log_likelihood_survival(
        &self,
        x: &Data,
        times: &[f64],
        events: &[bool],
    ) -> Result<Vec<Vec<f64>>> {
        if !matches!(self.config.outcome, Outcome::Aft(_)) {
            return Err(self.not_applicable("log_likelihood_survival"));
        }
        if times.len() != x.n_rows() {
            return Err(Error::RowCountMismatch {
                y_len: times.len(),
                x_rows: x.n_rows(),
            });
        }
        if events.len() != times.len() {
            return Err(Error::EventCountMismatch {
                events: events.len(),
                times: times.len(),
            });
        }
        if let Some(row) = times.iter().position(|&t| !(t.is_finite() && t > 0.0)) {
            return Err(Error::InvalidSurvivalTime { row });
        }
        let per_draw = self.predict_draws(x)?;
        let variances = self.predict_variance(x)?;
        let ln_2pi = maths::ln(2.0 * std::f64::consts::PI);
        let log_times: Vec<f64> = times.iter().map(|&t| maths::ln(t)).collect();
        Ok(per_draw
            .iter()
            .zip(&variances)
            .map(|(fits, variance)| {
                log_times
                    .iter()
                    .zip(events)
                    .zip(fits.iter().zip(variance))
                    .map(|((&v, &event), (&fit, &var))| {
                        if event {
                            let z = (v - fit) * (v - fit) / var;
                            -0.5 * (ln_2pi + maths::ln(var) + z)
                        } else {
                            maths::ln(maths::normal_cdf((fit - v) / var.sqrt()))
                        }
                    })
                    .collect()
            })
            .collect())
    }

    /// sigma per kept draw under a model with a global sampled sigma^2
    /// (the Gaussian and tobit models), caller scale: sqrt(sigma^2) times
    /// the training range of the response. Empty under the probit model
    /// (unit latent variance) and under a variance ensemble
    /// ([`predict_variance`](Self::predict_variance) gives s^2(x)).
    pub fn sigma(&self) -> Vec<f64> {
        let range = self.scaler.y_range();
        self.posterior
            .sigma_sq()
            .iter()
            .map(|s| s.sqrt() * range)
            .collect()
    }

    /// The probit offset c; 0 under the other models.
    fn offset(&self) -> f64 {
        self.config.offset().unwrap_or(0.0)
    }

    /// Mean number of cells per mean tessellation, one value per kept draw.
    pub fn cell_counts(&self) -> Vec<f64> {
        self.posterior
            .tessellations()
            .iter()
            .map(|draw| draw.iter().map(|t| t.n_cells() as f64).sum::<f64>() / draw.len() as f64)
            .collect()
    }

    /// Mean number of active covariates per mean tessellation, one value
    /// per kept draw.
    pub fn dimension_counts(&self) -> Vec<f64> {
        self.posterior
            .tessellations()
            .iter()
            .map(|draw| draw.iter().map(|t| t.n_dims() as f64).sum::<f64>() / draw.len() as f64)
            .collect()
    }

    /// Share of active mean-tessellation dimensions over all kept draws
    /// that fall on each covariate; sums to 1 (Chipman, George and
    /// McCulloch 2010, s. 5.1).
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

    /// The configuration the model was fitted with, omega resolved (and the
    /// offset, under the probit model).
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Fit-time warnings.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// Root mean square of the posterior-mean prediction against the
    /// training response, caller scale; under the probit model the root
    /// Brier score of the predicted probabilities against the labels;
    /// under the tobit model the target is the observed response,
    /// censored rows at their limits.
    pub fn in_sample_rmse(&self) -> f64 {
        self.in_sample_rmse
    }
}

fn column_means(per_draw: &[Vec<f64>], n: usize) -> Vec<f64> {
    let n_draws = per_draw.len() as f64;
    let mut means = vec![0.0; n];
    for draw in per_draw {
        for (mean, value) in means.iter_mut().zip(draw) {
            *mean += value;
        }
    }
    for mean in &mut means {
        *mean /= n_draws;
    }
    means
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

    fn data() -> (Data, Vec<f64>) {
        let n = 30;
        let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = xs.iter().map(|&v| 3.0 * v * v - v).collect();
        (Data::new(xs, n, 1).unwrap(), y)
    }

    fn small() -> Config {
        Config::new().with_m(10).with_burn_in(20).with_draws(30)
    }

    fn fitted() -> (Fitted, Data, Vec<f64>) {
        let (x, y) = data();
        (fit(&small(), &x, &y, 42).unwrap(), x, y)
    }

    fn fitted_probit() -> (Fitted, Data, Vec<f64>) {
        let (x, y) = data();
        let labels: Vec<f64> = y.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect();
        let config = small().with_outcome(Outcome::probit());
        (fit(&config, &x, &labels, 42).unwrap(), x, labels)
    }

    fn fitted_heteroscedastic() -> (Fitted, Data, Vec<f64>) {
        let (x, y) = data();
        let config = small().with_m_var(5);
        (fit(&config, &x, &y, 42).unwrap(), x, y)
    }

    #[test]
    fn prediction_surface_shapes_and_order() {
        let (model, x, y) = fitted();
        assert_eq!(model.model_name(), "gaussian");
        let mean = model.predict(&x).unwrap();
        let draws = model.predict_draws(&x).unwrap();
        assert_eq!(mean.len(), 30);
        assert_eq!((draws.len(), draws[0].len()), (30, 30));
        assert_eq!(model.predict_latent(&x).unwrap(), draws);
        let variance = model.predict_variance(&x).unwrap();
        let sigma = model.sigma();
        for (row, &s) in variance.iter().zip(&sigma) {
            assert!(row.iter().all(|&v| (v - s * s).abs() < 1e-12 * s * s));
        }
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
    fn probit_surface() {
        let (model, x, labels) = fitted_probit();
        assert_eq!(model.model_name(), "probit");
        let probs = model.predict(&x).unwrap();
        assert!(probs.iter().all(|p| (0.0..=1.0).contains(p)));
        // The mean probability over the draws is Phi of the latent, draw by
        // draw.
        let latent = model.predict_latent(&x).unwrap();
        let draws = model.predict_draws(&x).unwrap();
        for (l, d) in latent.iter().flatten().zip(draws.iter().flatten()) {
            assert!((maths::normal_cdf(*l) - d).abs() < 1e-15);
        }
        let offset = model.config().offset().unwrap();
        assert!(offset.is_finite());
        assert!(model.sigma().is_empty());
        assert!(matches!(
            model.predict_variance(&x),
            Err(Error::NotApplicable { ref method, ref model }) if method == "predict_variance" && model == "probit"
        ));
        assert!(matches!(
            model.prediction_interval(&x, 0.9),
            Err(Error::NotApplicable { .. })
        ));
        let ci = model.credible_interval(&x, 0.9).unwrap();
        assert!(ci.iter().all(|c| 0.0 <= c.lower && c.upper <= 1.0));
        let ll = model.log_likelihood(&x, &labels).unwrap();
        assert!(ll.iter().flatten().all(|v| v.is_finite() && *v <= 0.0));
        assert_eq!(
            model.log_likelihood(&x, &vec![0.5; 30]).unwrap_err(),
            Error::InvalidLabel { row: 0 }
        );
        // Root Brier score: strictly below the constant-probability score.
        let share = labels.iter().sum::<f64>() / 30.0;
        assert!(model.in_sample_rmse() < (share * (1.0 - share)).sqrt());
    }

    #[test]
    fn heteroscedastic_surface() {
        let (model, x, y) = fitted_heteroscedastic();
        assert_eq!(model.model_name(), "heteroscedastic");
        assert!(model.sigma().is_empty());
        let variance = model.predict_variance(&x).unwrap();
        assert_eq!((variance.len(), variance[0].len()), (30, 30));
        assert!(variance.iter().flatten().all(|v| v.is_finite() && *v > 0.0));
        // s^2 is the product of the variance cell values times the range squared.
        let draw0 = &model.posterior().variance_tessellations()[0];
        let x_scaled = model.scaler().scale_x(&x);
        let g = model.geometry().unwrap();
        let product: f64 = draw0
            .iter()
            .map(|t| t.value_at(x_scaled.row(4), &g))
            .product();
        let range = model.scaler().y_range();
        assert!((variance[0][4] - product * range * range).abs() < 1e-12 * variance[0][4]);
        let pi = model.prediction_interval(&x, 0.9).unwrap();
        let ci = model.credible_interval(&x, 0.9).unwrap();
        for (c, p) in ci.iter().zip(&pi) {
            assert!(p.lower <= c.lower && c.upper <= p.upper);
        }
        let ll = model.log_likelihood(&x, &y).unwrap();
        assert!(ll.iter().flatten().all(|v| v.is_finite()));
        let expected = -0.5
            * ((2.0 * std::f64::consts::PI * variance[0][4]).ln()
                + (y[4] - model.predict_draws(&x).unwrap()[0][4]).powi(2) / variance[0][4]);
        assert!((ll[0][4] - expected).abs() < 1e-12);
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
        for (model, x, _) in [fitted(), fitted_probit(), fitted_heteroscedastic()] {
            let json = serde_json::to_string(&model).unwrap();
            let loaded: Fitted = serde_json::from_str(&json).unwrap();
            assert_eq!(loaded, model);
            assert_eq!(loaded.predict(&x).unwrap(), model.predict(&x).unwrap());
            let corrupt = json.replace("\"in_sample_rmse\":", "\"in_sample_rmse\":1e999,\"x\":");
            assert!(serde_json::from_str::<Fitted>(&corrupt).is_err());
        }
    }

    #[test]
    fn loading_validates_the_model_specific_parts() {
        let (gaussian, _, _) = fitted();
        let (probit, _, _) = fitted_probit();
        let (hetero, _, _) = fitted_heteroscedastic();
        let json = |m: &Fitted| serde_json::to_string(m).unwrap();
        // A Gaussian payload relabelled probit: sigma draws present, no offset.
        let gaussian_outcome = r#""outcome":{"gaussian":{"nu":6.0,"q":0.85}}"#;
        let relabelled =
            json(&gaussian).replace(gaussian_outcome, r#""outcome":{"probit":{"offset":0.0}}"#);
        assert_ne!(relabelled, json(&gaussian));
        assert!(serde_json::from_str::<Fitted>(&relabelled).is_err());
        // A probit payload relabelled Gaussian: no sigma draws.
        let offset = probit.config().offset().unwrap();
        let probit_outcome = format!(r#""outcome":{{"probit":{{"offset":{offset:?}}}}}"#);
        let relabelled = json(&probit).replace(&probit_outcome, gaussian_outcome);
        assert_ne!(relabelled, json(&probit));
        assert!(serde_json::from_str::<Fitted>(&relabelled).is_err());
        // A heteroscedastic payload with the wrong variance count.
        let wrong = json(&hetero).replace("\"tessellations\":5", "\"tessellations\":6");
        assert_ne!(wrong, json(&hetero));
        assert!(serde_json::from_str::<Fitted>(&wrong).is_err());
        // A Gaussian payload without the variance field (a pre-0.2 save).
        let old = json(&gaussian).replace(",\"variance_tessellations\":[]", "");
        assert_ne!(old, json(&gaussian));
        assert_eq!(serde_json::from_str::<Fitted>(&old).unwrap(), gaussian);
    }

    #[test]
    fn mixture_quantile_of_one_normal() {
        let q = mixture_quantile(&[1.0], &[2.0], 0.975);
        assert!((q - (1.0 + 2.0 * 1.959_963_984_540_054)).abs() < 1e-9);
    }
}
