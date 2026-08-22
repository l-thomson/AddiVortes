//! The model discriminator shared by `Config`, `Sampler` and `Fitted`.

use std::str::FromStr;

use crate::error::{invalid, Error};
use crate::outcome::{RequiredData, Sigma2Mode};

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
///
/// Serialises as the snake-case name. The name of a variant behind the
/// `experimental` feature is rejected in a build without it with
/// [`Error::RequiresFeature`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Model {
    /// Gaussian noise with one global variance.
    #[default]
    Gaussian,
    /// Binary response through a probit link.
    Probit,
    /// Gaussian noise with a covariate-dependent variance.
    Heteroscedastic,
}

/// Names of the variants compiled only with the `experimental` feature.
#[cfg(not(feature = "experimental"))]
const EXPERIMENTAL: &[&str] = &[];

impl Model {
    /// What the model does with sigma^2; scale validity derives from this
    /// value, never from a per-model table.
    pub(crate) fn sigma2_mode(self) -> Sigma2Mode {
        match self {
            Model::Gaussian | Model::Heteroscedastic => Sigma2Mode::Sampled,
            Model::Probit => Sigma2Mode::Fixed(1.0),
        }
    }

    /// The response contract the model imposes at fit.
    pub(crate) fn required_data(self) -> RequiredData {
        match self {
            Model::Gaussian | Model::Heteroscedastic => RequiredData::Continuous,
            Model::Probit => RequiredData::Binary,
        }
    }

    /// Whether the model draws and stores a global sigma^2: a sampled
    /// scale that no variance ensemble carries instead.
    pub(crate) fn has_global_variance(self) -> bool {
        self.sigma2_mode().samples_global_sigma_sq() && !self.has_variance_ensemble()
    }

    /// Whether the model carries a variance ensemble.
    pub(crate) fn has_variance_ensemble(self) -> bool {
        matches!(self, Model::Heteroscedastic)
    }

    fn name(self) -> &'static str {
        match self {
            Model::Gaussian => "gaussian",
            Model::Probit => "probit",
            Model::Heteroscedastic => "heteroscedastic",
        }
    }
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Model {
    type Err = Error;

    /// # Errors
    ///
    /// [`Error::RequiresFeature`] for the name of a gated variant;
    /// [`Error::InvalidHyperparameter`] for any other unknown name.
    fn from_str(s: &str) -> Result<Self, Error> {
        match s {
            "gaussian" => Ok(Model::Gaussian),
            "probit" => Ok(Model::Probit),
            "heteroscedastic" => Ok(Model::Heteroscedastic),
            #[cfg(not(feature = "experimental"))]
            _ if EXPERIMENTAL.contains(&s) => Err(Error::RequiresFeature {
                item: format!("model `{s}`"),
                feature: "experimental",
            }),
            _ => Err(invalid("model", format!("unknown model `{s}`"))),
        }
    }
}

impl TryFrom<String> for Model {
    type Error = Error;

    fn try_from(s: String) -> Result<Self, Error> {
        s.parse()
    }
}

impl From<Model> for String {
    fn from(model: Model) -> Self {
        model.name().to_owned()
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
        assert_eq!("gaussian".parse::<Model>().unwrap(), Model::Gaussian);
        assert!(matches!(
            "cauchy".parse::<Model>(),
            Err(Error::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn scale_validity_derives_from_the_mode() {
        assert_eq!(Model::Gaussian.sigma2_mode(), Sigma2Mode::Sampled);
        assert_eq!(Model::Probit.sigma2_mode(), Sigma2Mode::Fixed(1.0));
        assert_eq!(Model::Heteroscedastic.sigma2_mode(), Sigma2Mode::Sampled);
        assert!(Model::Gaussian.has_global_variance());
        assert!(!Model::Probit.has_global_variance());
        assert!(!Model::Heteroscedastic.has_global_variance());
        assert!(Model::Heteroscedastic.has_variance_ensemble());
        assert!(!Model::Probit.sigma2_mode().permits_variance_ensemble());
        assert_eq!(Model::Probit.required_data(), RequiredData::Binary);
        assert_eq!(Model::Gaussian.required_data(), RequiredData::Continuous);
    }

    #[cfg(not(feature = "experimental"))]
    #[test]
    fn gated_names_name_the_feature() {
        for name in EXPERIMENTAL {
            assert!(matches!(
                name.parse::<Model>(),
                Err(Error::RequiresFeature {
                    feature: "experimental",
                    ..
                })
            ));
            let message = serde_json::from_str::<Model>(&format!("\"{name}\""))
                .unwrap_err()
                .to_string();
            assert!(message.contains("experimental"), "{message}");
        }
    }
}
