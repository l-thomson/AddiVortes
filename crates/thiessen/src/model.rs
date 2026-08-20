//! The model discriminator shared by `Config`, `Sampler` and `Fitted`.

/// The observation model. Selected on [`Config`](crate::Config) and stored
/// on the fitted model; the sampler's verbs and the prediction surface keep
/// their signatures and change meaning as documented per method.
///
/// - `Gaussian`: y = f(x) + e, e ~ N(0, sigma^2) (Stone and Gosling 2025).
/// - `Probit`: y in {0, 1}, P(y = 1 | x) = Phi(offset + f(x)), fitted by
///   Albert and Chib (1993) augmentation with unit latent variance.
/// - `Heteroscedastic`: y = f(x) + s(x) e, e ~ N(0, 1), s^2(x) the product
///   of `m_var` variance tessellations with inverse-gamma cell values
///   (the structure of HBART, Pratola et al. 2020).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Model {
    /// Gaussian noise with one global variance.
    #[default]
    Gaussian,
    /// Binary response through a probit link.
    Probit,
    /// Gaussian noise with a covariate-dependent variance.
    Heteroscedastic,
}

impl Model {
    /// Whether the model draws a global sigma^2.
    pub(crate) fn has_global_variance(self) -> bool {
        matches!(self, Model::Gaussian)
    }

    /// Whether the model carries a variance ensemble.
    pub(crate) fn has_variance_ensemble(self) -> bool {
        matches!(self, Model::Heteroscedastic)
    }
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Model::Gaussian => "gaussian",
            Model::Probit => "probit",
            Model::Heteroscedastic => "heteroscedastic",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_is_snake_case_and_defaults_to_gaussian() {
        assert_eq!(Model::default(), Model::Gaussian);
        assert_eq!(
            serde_json::to_string(&Model::Heteroscedastic).unwrap(),
            "\"heteroscedastic\""
        );
        assert_eq!(
            serde_json::from_str::<Model>("\"probit\"").unwrap(),
            Model::Probit
        );
        assert!(serde_json::from_str::<Model>("\"Probit\"").is_err());
        assert_eq!(Model::Probit.to_string(), "probit");
    }
}
