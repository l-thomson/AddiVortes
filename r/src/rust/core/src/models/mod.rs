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

#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod aft;
#[cfg(feature = "experimental")]
pub(crate) mod censoring;
pub mod gaussian;
pub mod heteroscedastic;
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod interval_censored;
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod laplace;
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod ordinal;
pub mod probit;
#[cfg(feature = "experimental")]
pub(crate) mod scale_mixture;
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub mod student_t;
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

/// Fit `n_chains` chains of the model the configuration names, seeded by
/// [`chain_seed`](crate::chain_seed)`(seed, k)` for k below `n_chains`,
/// and pool their draws in chain order.
///
/// # Arguments
///
/// As [`fit`]. The second value is the posterior mean at the rows of
/// `x`, the fitted values, from the one prediction pass that also gives
/// the pooled in-sample RMSE ([`Fitted::pool_samplers`]).
///
/// # Errors
///
/// [`Sampler::new`]; `InvalidHyperparameter` for `n_chains` of zero.
pub fn fit_chains(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    n_chains: usize,
) -> Result<(Fitted, Vec<f64>)> {
    fit_chains_with_threads(config, x, y, seed, n_chains, 1)
}

/// Fit as [`fit_chains`], the chains spread over at most `n_threads`
/// threads, and at most the parallelism available to the process
/// ([`Sampler::advance_all`]). The draws do not depend on
/// `n_threads`: each chain runs on one thread with its own generator. The
/// fitted model's [`threads`](Fitted::threads) is `n_threads`, so its
/// predictions use the same count.
///
/// # Errors
///
/// [`fit_chains`].
pub fn fit_chains_with_threads(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    n_chains: usize,
    n_threads: usize,
) -> Result<(Fitted, Vec<f64>)> {
    if n_chains == 0 {
        return Err(crate::error::invalid(
            "n_chains",
            "a fit needs at least one chain",
        ));
    }
    let mut samplers = (0..n_chains)
        .map(|k| Sampler::new(config, x, y, crate::chain_seed(seed, k)))
        .collect::<Result<Vec<_>>>()?;
    let schedule = &config.general_params;
    let mut chains: Vec<&mut Sampler> = samplers.iter_mut().collect();
    Sampler::advance_all(
        &mut chains,
        schedule.burn_in,
        schedule.draws,
        schedule.thinning,
        n_threads,
    );
    let (mut fitted, values) = Fitted::pool_samplers(samplers, x, y)?;
    fitted.set_threads(n_threads);
    Ok((fitted, values))
}

/// Fit the AFT model: as [`fit`], with the times and the event
/// indicator in place of a plain response ([`Outcome::Aft`](crate::Outcome::Aft)).
/// Experimental (`docs/experimental.md`).
///
/// # Arguments
///
/// `x` is n by p; `times` has n positive event or censoring times and
/// `events` one flag per row (true is an event, false right-censoring);
/// `seed` keys the chain RNG.
///
/// # Errors
///
/// [`Sampler::aft`].
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub fn fit_aft(
    config: &Config,
    x: &Data,
    times: &[f64],
    events: &[bool],
    seed: u64,
) -> Result<Fitted> {
    run_schedule(
        Sampler::aft(config, x, times, events, seed)?,
        config,
        &mut |_, _| {},
    )
}

/// Fit the interval-censored model: as [`fit`], with one pair of bounds
/// per row in place of a plain response
/// ([`Outcome::IntervalCensored`](crate::Outcome::IntervalCensored)).
/// Experimental (`docs/experimental.md`).
///
/// # Arguments
///
/// `x` is n by p; `lower` and `upper` hold n bounds each (an equal pair
/// is an exact value, an infinite endpoint one-sided censoring); `seed`
/// keys the chain RNG.
///
/// # Errors
///
/// [`Sampler::interval_censored`].
#[cfg(feature = "experimental")]
#[cfg_attr(docsrs, doc(cfg(feature = "experimental")))]
pub fn fit_interval_censored(
    config: &Config,
    x: &Data,
    lower: &[f64],
    upper: &[f64],
    seed: u64,
) -> Result<Fitted> {
    run_schedule(
        Sampler::interval_censored(config, x, lower, upper, seed)?,
        config,
        &mut |_, _| {},
    )
}

/// The sweep schedule shared by every model.
pub(crate) fn run(
    config: &Config,
    x: &Data,
    y: &[f64],
    seed: u64,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Fitted> {
    run_schedule(Sampler::new(config, x, y, seed)?, config, progress)
}

/// Run a constructed sampler through the configured schedule and finish
/// it.
fn run_schedule(
    mut sampler: Sampler,
    config: &Config,
    progress: &mut dyn FnMut(usize, usize),
) -> Result<Fitted> {
    advance(&mut sampler, config, progress);
    sampler.finish()
}

/// Advance a constructed sampler through the configured schedule.
fn advance(sampler: &mut Sampler, config: &Config, progress: &mut dyn FnMut(usize, usize)) {
    let schedule = config.general_params.clone();
    let total = schedule.burn_in + schedule.draws * schedule.thinning;
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
}
