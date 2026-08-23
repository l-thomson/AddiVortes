//! The observation models, one module each: the model statement, its
//! priors, what is fixed rather than estimated, the correspondence of its
//! parameters with the paper and with the BART-family reference
//! implementation, and the prediction semantics of the fitted model. The
//! model is selected by [`Config::outcome`]; every model runs through
//! [`fit`] and the [`Sampler`].

use crate::config::Config;
use crate::data::Data;
use crate::error::Result;
use crate::fitted::Fitted;
use crate::sampler::Sampler;

pub mod gaussian;
pub mod heteroscedastic;
pub mod probit;
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod tobit;

/// Fit the model the configuration names: validate, run `burn_in` sweeps,
/// then keep every `thinning`-th of the next `draws * thinning` sweeps.
///
/// # Arguments
///
/// `x` is n by p, `y` has n rows, both on the caller's scale (labels in
/// {0, 1} under the probit model); `seed` keys the chain RNG.
///
/// # Errors
///
/// [`Sampler::new`].
pub fn fit(config: &Config, x: &Data, y: &[f64], seed: u64) -> Result<Fitted> {
    fit_with_progress(config, x, y, seed, |_, _| {})
}

/// Fit as [`fit`], calling `progress(completed, total)` after every sweep.
///
/// # Arguments
///
/// As [`fit`]. `total` is `burn_in + draws * thinning`, `completed` counts
/// sweeps from one, and the draws do not depend on `progress`.
///
/// # Errors
///
/// [`Sampler::new`].
pub fn fit_with_progress(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    mut progress: impl FnMut(usize, usize),
) -> Result<Fitted> {
    run(config, x, y, seed, &mut progress)
}

/// The sweep schedule shared by every model.
pub(crate) fn run(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Fitted> {
    let schedule = config.general_params.clone();
    let total = schedule.burn_in + schedule.draws * schedule.thinning;
    let mut sampler = Sampler::new(config, x, y, seed)?;
    let mut completed = 0;
    for _ in 0..schedule.burn_in {
        sampler.step();
        completed += 1;
        progress(completed, total);
    }
    for _ in 0..schedule.draws {
        for _ in 0..schedule.thinning {
            sampler.step();
            completed += 1;
            progress(completed, total);
        }
        sampler.keep();
    }
    sampler.finish()
}
