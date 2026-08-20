//! Model configuration (`Config`): plain data with serde, `Default`,
//! consuming `with_*` setters and a data-free `validate()`.

use crate::error::{invalid, Result};
use crate::model::Model;

/// Configuration of an AddiVortes fit: the observation model, the
/// hyperparameters of Stone and Gosling (2025), s. 2, and the sweep
/// schedule that `fit` runs.
///
/// Every field has a default; unset JSON fields take it; unknown fields are
/// rejected. The seed is not part of the configuration; it is an argument
/// to [`fit`](crate::fit) and [`Sampler::new`](crate::Sampler::new).
///
/// Setters never panic or clamp; [`validate`](Config::validate) checks
/// every field and `fit` calls it first. Fields a model does not use
/// (`offset` outside the probit model, `m_var` outside the heteroscedastic
/// model) are not validated and have no effect.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default, deny_unknown_fields)]
#[non_exhaustive]
pub struct Config {
    /// The observation model. Default [`Model::Gaussian`].
    pub model: Model,
    /// Ensemble size m of the mean function. Default 200.
    pub m: usize,
    /// sigma^2 prior degrees of freedom nu. Default 6. The heteroscedastic
    /// model requires nu > 2.
    pub nu: f64,
    /// sigma^2 prior calibration quantile q, Pr(sigma < sigma_hat) = q.
    /// Default 0.85.
    pub q: f64,
    /// Cell-mean prior spread k: sigma_mu = 0.5 / (k sqrt(m)) on the
    /// response scaled to [-0.5, 0.5] (Gaussian and heteroscedastic
    /// models), 3 / (k sqrt(m)) on the latent scale (probit model; Chipman,
    /// George and McCulloch 2010, s. 4). Default 3.
    pub k: f64,
    /// Centre-coordinate prior and proposal standard deviation sigma_c
    /// (scaled space). Default 0.8.
    pub sigma_c: f64,
    /// Dimension-count prior parameter omega; omega / p is the prior
    /// probability of including a covariate. `None` resolves to min(3, p)
    /// at fit. Must satisfy 0 < omega <= p; at omega = p the dimension count
    /// saturates at p.
    pub omega: Option<f64>,
    /// Cell-count prior rate lambda_c: b - 1 ~ Poisson(lambda_c). Default 5,
    /// following AddiVortes >= 0.6.8; the paper reports 25
    /// ([`Config::paper`]).
    pub lambda_c: f64,
    /// Burn-in sweeps discarded by `fit`. Default 200.
    pub burn_in: usize,
    /// Posterior draws kept by `fit`. Default 1000.
    pub draws: usize,
    /// Thinning interval: `fit` keeps every `thinning`-th sweep after
    /// burn-in. Default 1.
    pub thinning: usize,
    /// Sample from the prior: the likelihood is switched off, so the chain
    /// draws sigma^2, the tessellations and the cell means from the prior
    /// and `predict` on the fitted model gives prior predictive draws (brms
    /// `sample_prior = "only"`). The response still fixes the scaling and
    /// the lambda calibration, so the prior sampled is the prior a fit on
    /// the same data would use, and the empty-cell rejection stays, so
    /// tessellation draws are truncated to those whose cells all hold a
    /// training row. Default false.
    pub prior_only: bool,
    /// Probit model only: the offset c in P(y = 1 | x) = Phi(c + f(x)).
    /// `None` resolves to Phi^-1(ybar) at fit (BART `binaryOffset`); the
    /// resolved value is stored on the fitted model. Default `None`.
    pub offset: Option<f64>,
    /// Heteroscedastic model only: the number m' of variance
    /// tessellations. Default 40.
    pub m_var: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: Model::Gaussian,
            m: 200,
            nu: 6.0,
            q: 0.85,
            k: 3.0,
            sigma_c: 0.8,
            omega: None,
            lambda_c: 5.0,
            burn_in: 200,
            draws: 1000,
            thinning: 1,
            prior_only: false,
            offset: None,
            m_var: 40,
        }
    }
}

impl Config {
    /// The defaults ([`Default`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// The paper's settings: the defaults with lambda_c = 25 (Stone and
    /// Gosling 2025, s. 2.3).
    pub fn paper() -> Self {
        Self::default().with_lambda_c(25.0)
    }

    /// Ensemble size m.
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// sigma^2 prior degrees of freedom nu.
    #[must_use]
    pub fn with_nu(mut self, nu: f64) -> Self {
        self.nu = nu;
        self
    }

    /// sigma^2 prior calibration quantile q.
    #[must_use]
    pub fn with_q(mut self, q: f64) -> Self {
        self.q = q;
        self
    }

    /// Cell-mean prior spread k.
    #[must_use]
    pub fn with_k(mut self, k: f64) -> Self {
        self.k = k;
        self
    }

    /// Centre-coordinate standard deviation sigma_c.
    #[must_use]
    pub fn with_sigma_c(mut self, sigma_c: f64) -> Self {
        self.sigma_c = sigma_c;
        self
    }

    /// Dimension-count prior parameter omega.
    #[must_use]
    pub fn with_omega(mut self, omega: f64) -> Self {
        self.omega = Some(omega);
        self
    }

    /// Cell-count prior rate lambda_c.
    #[must_use]
    pub fn with_lambda_c(mut self, lambda_c: f64) -> Self {
        self.lambda_c = lambda_c;
        self
    }

    /// Burn-in sweeps.
    #[must_use]
    pub fn with_burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Posterior draws kept.
    #[must_use]
    pub fn with_draws(mut self, draws: usize) -> Self {
        self.draws = draws;
        self
    }

    /// Thinning interval.
    #[must_use]
    pub fn with_thinning(mut self, thinning: usize) -> Self {
        self.thinning = thinning;
        self
    }

    /// Prior-only sampling.
    #[must_use]
    pub fn with_prior_only(mut self, prior_only: bool) -> Self {
        self.prior_only = prior_only;
        self
    }

    /// The observation model.
    #[must_use]
    pub fn with_model(mut self, model: Model) -> Self {
        self.model = model;
        self
    }

    /// Probit offset c.
    #[must_use]
    pub fn with_offset(mut self, offset: f64) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Number of variance tessellations m'.
    #[must_use]
    pub fn with_m_var(mut self, m_var: usize) -> Self {
        self.m_var = m_var;
        self
    }

    /// The omega in force for p covariates: the field, or min(3, p).
    pub(crate) fn omega_for(&self, p: usize) -> f64 {
        self.omega.unwrap_or_else(|| 3.0_f64.min(p as f64))
    }

    /// Data-free validation of every field the model uses.
    ///
    /// # Errors
    ///
    /// `InvalidHyperparameter` naming the field. The omega <= p check needs
    /// the data and runs at the fit boundary.
    pub fn validate(&self) -> Result<()> {
        self.validate_shared()?;
        match self.model {
            Model::Gaussian => Ok(()),
            Model::Probit => match self.offset {
                Some(c) if !c.is_finite() => {
                    Err(invalid("offset", format!("must be finite, got {c}")))
                }
                _ => Ok(()),
            },
            Model::Heteroscedastic => {
                if self.m_var < 1 {
                    return Err(invalid("m_var", "must be at least 1"));
                }
                if self.nu <= 2.0 {
                    return Err(invalid(
                        "nu",
                        format!(
                            "must exceed 2 under the heteroscedastic model, got {}",
                            self.nu
                        ),
                    ));
                }
                Ok(())
            }
        }
    }

    fn validate_shared(&self) -> Result<()> {
        let positive = |name: &str, value: f64| -> Result<()> {
            if value.is_finite() && value > 0.0 {
                Ok(())
            } else {
                Err(invalid(
                    name,
                    format!("must be finite and positive, got {value}"),
                ))
            }
        };
        if self.m < 1 {
            return Err(invalid("m", "must be at least 1"));
        }
        positive("nu", self.nu)?;
        if !(self.q.is_finite() && self.q > 0.0 && self.q < 1.0) {
            return Err(invalid(
                "q",
                format!("must be in the open interval (0, 1), got {}", self.q),
            ));
        }
        positive("k", self.k)?;
        positive("sigma_c", self.sigma_c)?;
        if let Some(omega) = self.omega {
            positive("omega", omega)?;
        }
        positive("lambda_c", self.lambda_c)?;
        if self.draws < 1 {
            return Err(invalid("draws", "must be at least 1"));
        }
        if self.thinning < 1 {
            return Err(invalid("thinning", "must be at least 1"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;

    #[test]
    fn defaults_validate() {
        assert!(Config::default().validate().is_ok());
        assert_eq!(Config::paper().lambda_c, 25.0);
    }

    #[test]
    fn every_field_is_checked() {
        let rejects = |config: Config, field: &str| {
            assert!(matches!(
                config.validate(),
                Err(Error::InvalidHyperparameter { ref name, .. }) if name == field
            ));
        };
        rejects(Config::new().with_m(0), "m");
        rejects(Config::new().with_nu(0.0), "nu");
        rejects(Config::new().with_q(1.0), "q");
        rejects(Config::new().with_k(f64::NAN), "k");
        rejects(Config::new().with_sigma_c(-1.0), "sigma_c");
        rejects(Config::new().with_omega(0.0), "omega");
        rejects(Config::new().with_lambda_c(f64::INFINITY), "lambda_c");
        rejects(Config::new().with_draws(0), "draws");
        rejects(Config::new().with_thinning(0), "thinning");
    }

    #[test]
    fn model_specific_fields_are_checked_only_under_their_model() {
        let rejects = |config: Config, field: &str| {
            assert!(matches!(
                config.validate(),
                Err(Error::InvalidHyperparameter { ref name, .. }) if name == field
            ));
        };
        let probit = Config::new().with_model(Model::Probit);
        let hetero = Config::new().with_model(Model::Heteroscedastic);
        rejects(probit.clone().with_offset(f64::NAN), "offset");
        rejects(hetero.clone().with_m_var(0), "m_var");
        rejects(hetero.clone().with_nu(2.0), "nu");
        assert!(probit.with_offset(0.3).validate().is_ok());
        assert!(hetero.with_m_var(1).validate().is_ok());
        assert!(Config::new()
            .with_offset(f64::NAN)
            .with_m_var(0)
            .validate()
            .is_ok());
        assert!(Config::new().with_nu(1.0).validate().is_ok());
    }

    #[test]
    fn model_serialises_in_snake_case_and_defaults_to_gaussian() {
        let parsed: Config = serde_json::from_str(r#"{"m": 7}"#).unwrap();
        assert_eq!(parsed.model, Model::Gaussian);
        let parsed: Config =
            serde_json::from_str(r#"{"model": "heteroscedastic", "m_var": 5}"#).unwrap();
        assert_eq!(parsed.model, Model::Heteroscedastic);
        assert_eq!(parsed.m_var, 5);
        let json = serde_json::to_string(&Config::new().with_model(Model::Probit)).unwrap();
        assert!(json.contains("\"model\":\"probit\""));
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
    fn serde_round_trip_partial_and_unknown_field() {
        let config = Config::new().with_m(20).with_omega(1.5);
        let json = serde_json::to_string(&config).unwrap();
        let back: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back, config);
        let partial: Config = serde_json::from_str(r#"{"m": 7}"#).unwrap();
        assert_eq!(partial, Config::new().with_m(7));
        assert!(serde_json::from_str::<Config>(r#"{"lambda_C": 5}"#).is_err());
    }
}
