//! Model configuration (`Config`): the outcome model and the parameter
//! groups, plain data with serde, `Default`, consuming `with_*` setters
//! and a data-free `validate()`.

use crate::error::{invalid, Result};
use crate::geometry::Metric;
use crate::outcome::{RequiredData, Sigma2Mode};

/// The observation model and its own parameters, including the sigma^2
/// settings: what generated y, as opposed to the ensembles that describe
/// its average and spread.
///
/// Serialises externally tagged in snake case, so a JSON configuration
/// reads `{"outcome": {"probit": {}}}`. Validity is derived from the
/// outcome's scale mode: a variance ensemble may attach exactly where
/// sigma^2 is sampled.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// y = f(x) + e, e ~ N(0, sigma^2) (Stone and Gosling 2025). With
    /// `variance_params.num_tessellations` above 0 the spread is the
    /// variance ensemble's product s^2(x) in place of the global sigma^2
    /// (H-AddiVortes).
    Gaussian(GaussianParams),
    /// y in {0, 1} with P(y = 1 | x) = Phi(c + f(x)), fitted by Albert
    /// and Chib (1993) augmentation with unit latent variance (Binary
    /// AddiVortes).
    Probit(ProbitParams),
}

impl Default for Outcome {
    fn default() -> Self {
        Outcome::Gaussian(GaussianParams::default())
    }
}

impl Outcome {
    /// The Gaussian outcome with its default sigma^2 prior.
    pub fn gaussian() -> Self {
        Outcome::Gaussian(GaussianParams::default())
    }

    /// The probit outcome with the offset resolved from the data at fit.
    pub fn probit() -> Self {
        Outcome::Probit(ProbitParams::default())
    }

    /// What the outcome does with sigma^2; scale validity derives from
    /// this value, never from a per-outcome table.
    pub(crate) fn sigma2_mode(&self) -> Sigma2Mode {
        match self {
            Outcome::Gaussian(_) => Sigma2Mode::Sampled,
            Outcome::Probit(_) => Sigma2Mode::Fixed(1.0),
        }
    }

    /// The response contract the outcome imposes at fit.
    pub(crate) fn required_data(&self) -> RequiredData {
        match self {
            Outcome::Gaussian(_) => RequiredData::Continuous,
            Outcome::Probit(_) => RequiredData::Binary,
        }
    }
}

/// The Gaussian outcome's parameters: the sigma^2 prior, folded onto the
/// outcome because they are facts about the observation model.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GaussianParams {
    /// sigma^2 prior degrees of freedom nu. Default 6. A variance
    /// ensemble requires nu > 2.
    pub nu: f64,
    /// sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.
    /// Default 0.85.
    pub q: f64,
}

impl Default for GaussianParams {
    fn default() -> Self {
        Self { nu: 6.0, q: 0.85 }
    }
}

/// The probit outcome's parameters.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ProbitParams {
    /// The offset c in P(y = 1 | x) = Phi(c + f(x)). `None` resolves to
    /// Phi^-1(ybar) at fit (the BART package's `binaryOffset`); the
    /// resolved value is stored on the fitted model. Default `None`.
    pub offset: Option<f64>,
}

/// One term group: the ensemble describing the average (`mean_params`) or
/// the spread (`variance_params`). Written once and instantiated per
/// slot, so a new option reaches both ensembles at once.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TermParams {
    /// Number of tessellations in the ensemble. `None` resolves at fit to
    /// 200 on the mean slot and to 0 on the variance slot; 0 on the
    /// variance slot is a constant spread, above 0 is H-AddiVortes (the
    /// paper's count is 40).
    pub num_tessellations: Option<usize>,
    /// Cell-value prior spread k: sigma_mu = w / (k sqrt m) with the
    /// half-width w the outcome model owns. Default 3. The variance
    /// ensemble's inverse-gamma cells do not use it.
    pub k: f64,
    /// Cell-count prior rate lambda_c: b - 1 ~ Poisson(lambda_c). Default
    /// 5, following AddiVortes >= 0.6.8; the paper reports 25
    /// ([`Config::paper`]).
    pub lambda_c: f64,
    /// The covariate space: the metric of each column and the
    /// centre-coordinate law's scale.
    pub geometry: GeometryParams,
    /// Which covariates a tessellation may use.
    pub structure: StructureParams,
    /// The within-cell response surface.
    pub cell: CellParams,
}

impl Default for TermParams {
    fn default() -> Self {
        Self {
            num_tessellations: None,
            k: 3.0,
            lambda_c: 5.0,
            geometry: GeometryParams::default(),
            structure: StructureParams::default(),
            cell: CellParams::default(),
        }
    }
}

impl TermParams {
    /// The count in force on the mean slot: the field, or 200.
    pub(crate) fn mean_tessellations(&self) -> usize {
        self.num_tessellations.unwrap_or(200)
    }

    /// The count in force on the variance slot: the field, or 0.
    pub(crate) fn variance_tessellations(&self) -> usize {
        self.num_tessellations.unwrap_or(0)
    }
}

/// The covariate space of one term group.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeometryParams {
    /// The metric of each covariate column, one entry per column in
    /// column order; empty, the default, is [`Metric::Euclidean`] on
    /// every column. Checked against the design at fit.
    pub metric: Vec<Metric>,
    /// Centre-coordinate prior and proposal standard deviation sigma_c
    /// (scaled space). Default 0.8.
    pub sigma_c: f64,
    /// The precision matrix of the Mahalanobis metric, row-major p x p
    /// over the encoded design; required exactly when a column's metric
    /// is [`Metric::Mahalanobis`]. Checked at fit: symmetric, positive
    /// definite. The active-subspace distance uses the principal
    /// submatrix on the active columns, which is not the conditional
    /// precision. Experimental (`docs/experimental.md`).
    #[cfg(feature = "experimental")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub precision: Option<Vec<f64>>,
}

impl Default for GeometryParams {
    fn default() -> Self {
        Self {
            metric: Vec::new(),
            sigma_c: 0.8,
            #[cfg(feature = "experimental")]
            precision: None,
        }
    }
}

/// The covariate-inclusion prior of one term group.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StructureParams {
    /// Dimension-count prior parameter omega; omega / p is the prior
    /// probability of including a covariate. `None` resolves to
    /// min(3, p) at fit. Must satisfy 0 < omega <= p; at omega = p the
    /// dimension count saturates at p.
    pub omega: Option<f64>,
}

/// The within-cell response surface of one term group. Carries no options
/// yet: every cell holds one constant value, the paper's basis.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CellParams {}

/// The sweep schedule and everything that belongs to no ensemble.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneralParams {
    /// Burn-in sweeps discarded by `fit`. Default 200.
    pub burn_in: usize,
    /// Posterior draws kept by `fit`. Default 1000.
    pub draws: usize,
    /// Thinning interval: `fit` keeps every `thinning`-th sweep after
    /// burn-in. Default 1.
    pub thinning: usize,
    /// Sample from the prior: the likelihood is switched off, so the
    /// chain draws sigma^2, the tessellations and the cell values from
    /// the prior and `predict` on the fitted model gives prior predictive
    /// draws (brms `sample_prior = "only"`). The response still fixes the
    /// scaling and the lambda calibration, so the prior sampled is the
    /// prior a fit on the same data would use, and the empty-cell
    /// rejection stays. Default false.
    pub prior_only: bool,
}

impl Default for GeneralParams {
    fn default() -> Self {
        Self {
            burn_in: 200,
            draws: 1000,
            thinning: 1,
            prior_only: false,
        }
    }
}

/// Configuration of an AddiVortes fit: the outcome model, one term group
/// per ensemble, and the sweep schedule (Stone and Gosling 2025, s. 2).
///
/// Every field has a default; unset JSON fields take it; unknown fields
/// are rejected. The seed is not part of the configuration; it is an
/// argument to [`fit`](crate::fit) and [`Sampler::new`](crate::Sampler::new).
///
/// Setters never panic or clamp; [`validate`](Config::validate) checks
/// every field in force and `fit` calls it first. The geometry and
/// structure setters write both slots, since the ensembles must declare
/// identical geometry while per-ensemble geometry awaits its
/// identification argument.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "ConfigParts")]
#[non_exhaustive]
pub struct Config {
    /// The observation model, carrying its own parameters and its sigma^2
    /// settings.
    pub outcome: Outcome,
    /// The ensemble describing the average.
    pub mean_params: TermParams,
    /// The ensemble describing the spread; a resolved count of 0, the
    /// default, is a constant spread.
    pub variance_params: TermParams,
    /// Burn-in, draws, thinning, prior-only sampling.
    pub general_params: GeneralParams,
}

impl Config {
    /// The defaults ([`Default`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// The paper's settings: the defaults with lambda_c = 25 on both
    /// slots (Stone and Gosling 2025, s. 2.3).
    pub fn paper() -> Self {
        Self::default().with_lambda_c(25.0)
    }

    /// The observation model.
    #[must_use]
    pub fn with_outcome(mut self, outcome: Outcome) -> Self {
        self.outcome = outcome;
        self
    }

    /// Mean-ensemble size m.
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.mean_params.num_tessellations = Some(m);
        self
    }

    /// Variance-ensemble size m'; above 0 is H-AddiVortes.
    #[must_use]
    pub fn with_m_var(mut self, m_var: usize) -> Self {
        self.variance_params.num_tessellations = Some(m_var);
        self
    }

    /// sigma^2 prior degrees of freedom nu of the Gaussian outcome; no
    /// effect under another outcome.
    #[must_use]
    pub fn with_nu(mut self, nu: f64) -> Self {
        if let Outcome::Gaussian(params) = &mut self.outcome {
            params.nu = nu;
        }
        self
    }

    /// sigma^2 prior calibration quantile q of the Gaussian outcome; no
    /// effect under another outcome.
    #[must_use]
    pub fn with_q(mut self, q: f64) -> Self {
        if let Outcome::Gaussian(params) = &mut self.outcome {
            params.q = q;
        }
        self
    }

    /// Probit offset c; no effect under another outcome.
    #[must_use]
    pub fn with_offset(mut self, offset: f64) -> Self {
        if let Outcome::Probit(params) = &mut self.outcome {
            params.offset = Some(offset);
        }
        self
    }

    /// Cell-value prior spread k, both slots.
    #[must_use]
    pub fn with_k(mut self, k: f64) -> Self {
        self.mean_params.k = k;
        self.variance_params.k = k;
        self
    }

    /// Cell-count prior rate lambda_c, both slots.
    #[must_use]
    pub fn with_lambda_c(mut self, lambda_c: f64) -> Self {
        self.mean_params.lambda_c = lambda_c;
        self.variance_params.lambda_c = lambda_c;
        self
    }

    /// Centre-coordinate standard deviation sigma_c, both slots.
    #[must_use]
    pub fn with_sigma_c(mut self, sigma_c: f64) -> Self {
        self.mean_params.geometry.sigma_c = sigma_c;
        self.variance_params.geometry.sigma_c = sigma_c;
        self
    }

    /// The metric of each covariate column, both slots.
    #[must_use]
    pub fn with_metric(mut self, metric: Vec<Metric>) -> Self {
        self.mean_params.geometry.metric.clone_from(&metric);
        self.variance_params.geometry.metric = metric;
        self
    }

    /// The Mahalanobis precision matrix, row-major p x p over the
    /// encoded design, both slots. Experimental.
    #[cfg(feature = "experimental")]
    #[must_use]
    pub fn with_precision(mut self, precision: Vec<f64>) -> Self {
        self.mean_params.geometry.precision = Some(precision.clone());
        self.variance_params.geometry.precision = Some(precision);
        self
    }

    /// Dimension-count prior parameter omega, both slots.
    #[must_use]
    pub fn with_omega(mut self, omega: f64) -> Self {
        self.mean_params.structure.omega = Some(omega);
        self.variance_params.structure.omega = Some(omega);
        self
    }

    /// Burn-in sweeps.
    #[must_use]
    pub fn with_burn_in(mut self, burn_in: usize) -> Self {
        self.general_params.burn_in = burn_in;
        self
    }

    /// Posterior draws kept.
    #[must_use]
    pub fn with_draws(mut self, draws: usize) -> Self {
        self.general_params.draws = draws;
        self
    }

    /// Thinning interval.
    #[must_use]
    pub fn with_thinning(mut self, thinning: usize) -> Self {
        self.general_params.thinning = thinning;
        self
    }

    /// Prior-only sampling.
    #[must_use]
    pub fn with_prior_only(mut self, prior_only: bool) -> Self {
        self.general_params.prior_only = prior_only;
        self
    }

    /// The fitted model's name: the outcome's, or "heteroscedastic" for
    /// the Gaussian outcome with a variance ensemble attached (the paper
    /// names of the shipped models).
    pub fn model_name(&self) -> &'static str {
        match (&self.outcome, self.variance_tessellations()) {
            (Outcome::Probit(_), _) => "probit",
            (Outcome::Gaussian(_), 0) => "gaussian",
            (Outcome::Gaussian(_), _) => "heteroscedastic",
        }
    }

    /// The mean-ensemble size in force: the field, or 200.
    pub fn mean_tessellations(&self) -> usize {
        self.mean_params.mean_tessellations()
    }

    /// The variance-ensemble size in force: the field, or 0.
    pub fn variance_tessellations(&self) -> usize {
        self.variance_params.variance_tessellations()
    }

    /// The probit offset, where the outcome has one.
    pub fn offset(&self) -> Option<f64> {
        match &self.outcome {
            Outcome::Probit(params) => params.offset,
            Outcome::Gaussian(_) => None,
        }
    }

    /// The resolved probit offset, written back at fit.
    pub(crate) fn set_offset(&mut self, offset: f64) {
        if let Outcome::Probit(params) = &mut self.outcome {
            params.offset = Some(offset);
        }
    }

    /// The Gaussian outcome's sigma^2 prior (nu, q); the defaults where
    /// the outcome carries no sigma^2 prior, on paths that never read
    /// them.
    pub(crate) fn sigma2_prior(&self) -> (f64, f64) {
        match &self.outcome {
            Outcome::Gaussian(params) => (params.nu, params.q),
            Outcome::Probit(_) => {
                let defaults = GaussianParams::default();
                (defaults.nu, defaults.q)
            }
        }
    }

    /// The omega in force for p covariates: the mean slot's field, or
    /// min(3, p).
    pub(crate) fn omega_for(&self, p: usize) -> f64 {
        self.mean_params
            .structure
            .omega
            .unwrap_or_else(|| 3.0_f64.min(p as f64))
    }

    /// The resolved counts and omega, written back at fit.
    pub(crate) fn resolve(&mut self, omega: f64) {
        self.mean_params.num_tessellations = Some(self.mean_tessellations());
        self.variance_params.num_tessellations = Some(self.variance_tessellations());
        self.mean_params.structure.omega = Some(omega);
        self.variance_params.structure.omega = Some(omega);
    }

    /// Data-free validation of every field in force.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter` naming the field. The omega <= p check and
    /// the `metric` length check need the data and run at the fit
    /// boundary.
    pub fn validate(&self) -> Result<()> {
        if self.mean_tessellations() < 1 {
            return Err(invalid(
                "mean_params.num_tessellations",
                "must be at least 1",
            ));
        }
        validate_term("mean_params", &self.mean_params)?;
        match &self.outcome {
            Outcome::Gaussian(params) => {
                positive("nu", params.nu)?;
                if !(params.q.is_finite() && params.q > 0.0 && params.q < 1.0) {
                    return Err(invalid(
                        "q",
                        format!("must be in the open interval (0, 1), got {}", params.q),
                    ));
                }
            }
            Outcome::Probit(params) => {
                if let Some(c) = params.offset {
                    if !c.is_finite() {
                        return Err(invalid("offset", format!("must be finite, got {c}")));
                    }
                }
            }
        }
        let m_var = self.variance_tessellations();
        if m_var > 0 {
            if !self.outcome.sigma2_mode().permits_variance_ensemble() {
                return Err(invalid(
                    "variance_params.num_tessellations",
                    "a variance ensemble needs a sampled sigma^2 to carry, and the \
                     probit latent scale is fixed at 1 for identification",
                ));
            }
            let (nu, _) = self.sigma2_prior();
            if nu <= 2.0 {
                return Err(invalid(
                    "nu",
                    format!("must exceed 2 under a variance ensemble, got {nu}"),
                ));
            }
            validate_term("variance_params", &self.variance_params)?;
            if self.variance_params.geometry != self.mean_params.geometry {
                return Err(invalid(
                    "variance_params.geometry",
                    "must equal mean_params.geometry; per-ensemble geometry awaits \
                     its identification argument",
                ));
            }
            if self.variance_params.structure != self.mean_params.structure {
                return Err(invalid(
                    "variance_params.structure",
                    "must equal mean_params.structure; per-ensemble structure awaits \
                     its identification argument",
                ));
            }
        }
        if self.general_params.draws < 1 {
            return Err(invalid("draws", "must be at least 1"));
        }
        if self.general_params.thinning < 1 {
            return Err(invalid("thinning", "must be at least 1"));
        }
        Ok(())
    }
}

/// The deserialisation shape of [`Config`]: the four groups plus catchers
/// for the flat pre-reshape field names, so a saved flat configuration
/// fails with an error naming the replacement rather than an unknown-field
/// message.
#[derive(Default, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
struct ConfigParts {
    outcome: Outcome,
    mean_params: TermParams,
    variance_params: TermParams,
    general_params: GeneralParams,
    #[serde(rename = "model")]
    legacy_model: Option<serde::de::IgnoredAny>,
    #[serde(rename = "m")]
    legacy_m: Option<serde::de::IgnoredAny>,
    #[serde(rename = "nu")]
    legacy_nu: Option<serde::de::IgnoredAny>,
    #[serde(rename = "q")]
    legacy_q: Option<serde::de::IgnoredAny>,
    #[serde(rename = "k")]
    legacy_k: Option<serde::de::IgnoredAny>,
    #[serde(rename = "sigma_c")]
    legacy_sigma_c: Option<serde::de::IgnoredAny>,
    #[serde(rename = "omega")]
    legacy_omega: Option<serde::de::IgnoredAny>,
    #[serde(rename = "lambda_c")]
    legacy_lambda_c: Option<serde::de::IgnoredAny>,
    #[serde(rename = "burn_in")]
    legacy_burn_in: Option<serde::de::IgnoredAny>,
    #[serde(rename = "draws")]
    legacy_draws: Option<serde::de::IgnoredAny>,
    #[serde(rename = "thinning")]
    legacy_thinning: Option<serde::de::IgnoredAny>,
    #[serde(rename = "prior_only")]
    legacy_prior_only: Option<serde::de::IgnoredAny>,
    #[serde(rename = "offset")]
    legacy_offset: Option<serde::de::IgnoredAny>,
    #[serde(rename = "m_var")]
    legacy_m_var: Option<serde::de::IgnoredAny>,
    #[serde(rename = "metric")]
    legacy_metric: Option<serde::de::IgnoredAny>,
}

impl TryFrom<ConfigParts> for Config {
    type Error = crate::error::Error;

    fn try_from(parts: ConfigParts) -> Result<Self> {
        let legacy = parts.legacy_model.is_some()
            || parts.legacy_m.is_some()
            || parts.legacy_nu.is_some()
            || parts.legacy_q.is_some()
            || parts.legacy_k.is_some()
            || parts.legacy_sigma_c.is_some()
            || parts.legacy_omega.is_some()
            || parts.legacy_lambda_c.is_some()
            || parts.legacy_burn_in.is_some()
            || parts.legacy_draws.is_some()
            || parts.legacy_thinning.is_some()
            || parts.legacy_prior_only.is_some()
            || parts.legacy_offset.is_some()
            || parts.legacy_m_var.is_some()
            || parts.legacy_metric.is_some();
        if legacy {
            return Err(invalid(
                "model",
                "the flat configuration is replaced by `outcome`, `mean_params`,                  `variance_params` and `general_params`; `model` is the `outcome`                  variant, with a variance ensemble as `variance_params.num_tessellations`",
            ));
        }
        Ok(Self {
            outcome: parts.outcome,
            mean_params: parts.mean_params,
            variance_params: parts.variance_params,
            general_params: parts.general_params,
        })
    }
}

fn positive(name: &str, value: f64) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(invalid(
            name,
            format!("must be finite and positive, got {value}"),
        ))
    }
}

fn validate_term(slot: &str, params: &TermParams) -> Result<()> {
    positive(&format!("{slot}.k"), params.k)?;
    positive(&format!("{slot}.lambda_c"), params.lambda_c)?;
    positive(&format!("{slot}.geometry.sigma_c"), params.geometry.sigma_c)?;
    #[cfg(feature = "experimental")]
    for kind in &params.geometry.metric {
        if let Metric::Minkowski { p } = *kind {
            if !(p.is_finite() && p >= 1.0) {
                return Err(invalid(
                    &format!("{slot}.geometry.metric"),
                    format!("the Minkowski order must be at least 1, got {p}"),
                ));
            }
        }
    }
    if let Some(omega) = params.structure.omega {
        positive(&format!("{slot}.structure.omega"), omega)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    fn rejects(config: Config, field: &str) {
        assert!(matches!(
            config.validate(),
            Err(Error::InvalidHyperparameter { ref name, .. }) if name == field
        ));
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn a_minkowski_order_below_one_is_rejected() {
        for p in [0.5, f64::NAN, f64::INFINITY] {
            rejects(
                Config::new().with_metric(vec![Metric::Minkowski { p }]),
                "mean_params.geometry.metric",
            );
        }
        assert!(Config::new()
            .with_metric(vec![Metric::Minkowski { p: 1.0 }])
            .validate()
            .is_ok());
    }

    #[test]
    fn defaults_validate_and_resolve() {
        let config = Config::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.mean_tessellations(), 200);
        assert_eq!(config.variance_tessellations(), 0);
        assert_eq!(config.model_name(), "gaussian");
        assert_eq!(Config::paper().mean_params.lambda_c, 25.0);
        assert_eq!(Config::paper().variance_params.lambda_c, 25.0);
    }

    #[test]
    fn every_field_in_force_is_checked() {
        rejects(Config::new().with_m(0), "mean_params.num_tessellations");
        rejects(Config::new().with_nu(0.0), "nu");
        rejects(Config::new().with_q(1.0), "q");
        rejects(Config::new().with_k(f64::NAN), "mean_params.k");
        rejects(
            Config::new().with_sigma_c(-1.0),
            "mean_params.geometry.sigma_c",
        );
        rejects(Config::new().with_omega(0.0), "mean_params.structure.omega");
        rejects(
            Config::new().with_lambda_c(f64::INFINITY),
            "mean_params.lambda_c",
        );
        rejects(Config::new().with_draws(0), "draws");
        rejects(Config::new().with_thinning(0), "thinning");
    }

    #[test]
    fn outcome_specific_fields_are_checked_only_under_their_outcome() {
        let probit = Config::new().with_outcome(Outcome::probit());
        rejects(probit.clone().with_offset(f64::NAN), "offset");
        assert!(probit.clone().with_offset(0.3).validate().is_ok());
        // Setters for another outcome's parameters have no effect.
        assert_eq!(probit.clone().with_nu(0.0), probit);
        assert_eq!(Config::new().with_offset(0.3), Config::new());
    }

    #[test]
    fn the_variance_ensemble_derives_its_validity_from_the_mode() {
        let hetero = Config::new().with_m_var(4);
        assert!(hetero.validate().is_ok());
        assert_eq!(hetero.model_name(), "heteroscedastic");
        rejects(hetero.clone().with_nu(2.0), "nu");
        rejects(
            Config::new().with_outcome(Outcome::probit()).with_m_var(4),
            "variance_params.num_tessellations",
        );
        let message = Config::new()
            .with_outcome(Outcome::probit())
            .with_m_var(4)
            .validate()
            .unwrap_err()
            .to_string();
        assert!(message.contains("identification"), "{message}");
        // A zero-count variance slot is not validated.
        assert!(Config::new()
            .with_outcome(Outcome::probit())
            .with_m_var(0)
            .validate()
            .is_ok());
    }

    #[test]
    fn the_slots_must_share_geometry_and_structure_when_attached() {
        let mut config = Config::new().with_m_var(4);
        config.variance_params.geometry.sigma_c = 0.5;
        rejects(config, "variance_params.geometry");
        let mut config = Config::new().with_m_var(4);
        config.variance_params.structure.omega = Some(1.0);
        rejects(config, "variance_params.structure");
        let mut detached = Config::new();
        detached.variance_params.geometry.sigma_c = 0.5;
        assert!(detached.validate().is_ok());
    }

    #[test]
    fn a_flat_configuration_names_the_replacement() {
        let err = serde_json::from_str::<Config>(r#"{"model": "gaussian", "m": 5}"#).unwrap_err();
        assert!(err.to_string().contains("`outcome`"), "{err}");
        let err = serde_json::from_str::<Config>(r#"{"m_var": 3}"#).unwrap_err();
        assert!(err.to_string().contains("variance_params"), "{err}");
        assert!(serde_json::from_str::<Config>(r#"{"unheard_of": 1}"#)
            .unwrap_err()
            .to_string()
            .contains("unheard_of"));
    }

    #[test]
    fn serde_round_trip_partial_and_unknown_field() {
        let config = Config::new().with_m(20).with_omega(1.5);
        let json = serde_json::to_string(&config).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back, config);
        let partial: Config =
            serde_json::from_str(r#"{"mean_params": {"num_tessellations": 7}}"#).unwrap();
        assert_eq!(partial, Config::new().with_m(7));
        // A partially specified variance slot keeps the slot's resolution.
        let partial: Config = serde_json::from_str(r#"{"variance_params": {"k": 2.0}}"#).unwrap();
        assert_eq!(partial.variance_tessellations(), 0);
        assert!(serde_json::from_str::<Config>(r#"{"lambda_C": 5}"#).is_err());
        assert!(serde_json::from_str::<Config>(r#"{"mean_params": {"lambda_C": 5}}"#).is_err());
    }

    #[test]
    fn outcome_serialises_externally_tagged_in_snake_case() {
        let probit: Config =
            serde_json::from_str(r#"{"outcome": {"probit": {"offset": 0.3}}}"#).unwrap();
        assert_eq!(probit.offset(), Some(0.3));
        let json = serde_json::to_string(&probit).unwrap();
        assert!(
            json.contains(r#""outcome":{"probit":{"offset":0.3}}"#),
            "{json}"
        );
        let gaussian = serde_json::to_string(&Config::new()).unwrap();
        assert!(
            gaussian.contains(r#""outcome":{"gaussian":{"nu":6.0,"q":0.85}}"#),
            "{gaussian}"
        );
        assert!(serde_json::from_str::<Config>(r#"{"outcome": {"cauchy": {}}}"#).is_err());
    }

    #[test]
    fn metric_serialises_by_name() {
        let config =
            Config::new().with_metric(vec![Metric::Euclidean, Metric::Spherical { sphere: 1 }]);
        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains(r#""metric":["euclidean",{"spherical":{"sphere":1}}]"#));
        assert_eq!(serde_json::from_str::<Config>(&json).unwrap(), config);
    }

    #[test]
    fn omega_default_is_min_three_p() {
        let config = Config::default();
        assert_eq!(config.omega_for(1), 1.0);
        assert_eq!(config.omega_for(2), 2.0);
        assert_eq!(config.omega_for(10), 3.0);
        assert_eq!(config.with_omega(0.5).omega_for(10), 0.5);
    }

    #[test]
    fn resolve_writes_the_counts_and_omega_back() {
        let mut config = Config::new().with_m_var(4);
        config.resolve(2.5);
        assert_eq!(config.mean_params.num_tessellations, Some(200));
        assert_eq!(config.variance_params.num_tessellations, Some(4));
        assert_eq!(config.mean_params.structure.omega, Some(2.5));
        assert_eq!(config.variance_params.structure.omega, Some(2.5));
    }
}
