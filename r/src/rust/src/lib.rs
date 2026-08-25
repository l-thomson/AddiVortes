//! extendr bindings over the `thiessen` core: the configuration as JSON,
//! the design as an R matrix, and the fitted model as a live handle behind
//! an external pointer.
//!
//! The configuration crosses the boundary as JSON so that the core's serde
//! representation stays the single definition of the field names, their
//! defaults and their validation. The fitted model stays on the Rust side:
//! methods take the handle, and the state is encoded to bytes only for
//! persistence, as MessagePack with field names, so the serde
//! representation stays the single definition there too. A double survives
//! the byte encoding bit for bit, which text cannot promise.

use extendr_api::prelude::*;
use extendr_api::Result;
use std::cell::RefCell;

fn core_error(error: thiessen::Error) -> Error {
    Error::Other(error.to_string())
}

fn json_error(error: serde_json::Error) -> Error {
    Error::Other(error.to_string())
}

fn config(json: &str) -> Result<thiessen::Config> {
    serde_json::from_str(json).map_err(json_error)
}

/// The fitted model, alive behind an external pointer. R releases it
/// through the pointer's finalizer.
struct FittedHandle {
    inner: thiessen::Fitted,
}

/// The design in row-major order; `RMatrix::data` is column-major.
fn design(x: &RMatrix<f64>) -> Result<thiessen::Data> {
    let (n_rows, n_cols) = (x.nrows(), x.ncols());
    let column_major = x.data();
    let mut values = Vec::with_capacity(n_rows * n_cols);
    for row in 0..n_rows {
        for col in 0..n_cols {
            values.push(column_major[col * n_rows + row]);
        }
    }
    thiessen::Data::new(values, n_rows, n_cols).map_err(core_error)
}

/// Draw-major rows as a matrix of one row per draw, as `dbarts` returns.
fn draw_matrix(draws: &[Vec<f64>]) -> RMatrix<f64> {
    let n_cols = draws.first().map_or(0, Vec::len);
    RMatrix::new_matrix(draws.len(), n_cols, |row, col| draws[row][col])
}

/// A whole non-negative double as the core's `u64` seed.
fn seed(seed: f64) -> Result<u64> {
    if !seed.is_finite() || seed < 0.0 || seed.fract() != 0.0 || seed > 9_007_199_254_740_992.0 {
        return Err(Error::Other(format!(
            "seed must be a whole number in [0, 2^53]; got {seed}"
        )));
    }
    Ok(seed as u64)
}

/// Whether the core was built with the `experimental` feature.
#[extendr]
fn core_experimental() -> bool {
    cfg!(feature = "experimental")
}

/// Version of the vendored core crate.
#[extendr]
fn core_version() -> &'static str {
    thiessen::VERSION
}

/// The core's default configuration as JSON.
#[extendr]
fn core_defaults() -> Result<String> {
    serde_json::to_string(&thiessen::Config::new()).map_err(json_error)
}

/// Validate a configuration without data.
#[extendr]
fn core_validate(config_json: &str) -> Result<()> {
    config(config_json)?.validate().map_err(core_error)
}

/// Whether `state` is a fitted-model pointer with a live address. A
/// pointer read back by `readRDS` deserialises with a null address, so a
/// caller checks here and restores from the payload before using it.
#[extendr]
fn core_state_is_live(state: Robj) -> bool {
    <&ExternalPtr<FittedHandle>>::try_from(&state).is_ok()
}

/// The fitted state as bytes, for persistence.
#[extendr]
fn core_state_payload(state: ExternalPtr<FittedHandle>) -> Result<Raw> {
    let bytes = rmp_serde::encode::to_vec_named(&state.inner)
        .map_err(|error| Error::Other(error.to_string()))?;
    Ok(Raw::from_bytes(&bytes))
}

/// A live fitted-model pointer from the bytes of `core_state_payload`.
#[extendr]
fn core_state_restore(payload: &[u8]) -> Result<ExternalPtr<FittedHandle>> {
    let inner: thiessen::Fitted =
        rmp_serde::from_slice(payload).map_err(|error| Error::Other(error.to_string()))?;
    Ok(ExternalPtr::new(FittedHandle { inner }))
}

/// The posterior mean at each row of `x`.
#[extendr]
fn core_predict(state: ExternalPtr<FittedHandle>, x: RMatrix<f64>) -> Result<Vec<f64>> {
    state.inner.predict(&design(&x)?).map_err(core_error)
}

/// Per-draw predictions: `kind` selects the quantity of `predict`, the mean
/// function, or the variance of y given f.
#[extendr]
fn core_predict_draws(
    state: ExternalPtr<FittedHandle>,
    x: RMatrix<f64>,
    kind: &str,
) -> Result<RMatrix<f64>> {
    let fitted = &state.inner;
    let data = design(&x)?;
    let draws = match kind {
        "draws" => fitted.predict_draws(&data),
        "latent" => fitted.predict_latent(&data),
        "variance" => fitted.predict_variance(&data),
        other => return Err(Error::Other(format!("unknown draw kind {other}"))),
    }
    .map_err(core_error)?;
    Ok(draw_matrix(&draws))
}

/// Central credible or posterior predictive interval, `n_rows` by 2.
#[extendr]
fn core_interval(
    state: ExternalPtr<FittedHandle>,
    x: RMatrix<f64>,
    kind: &str,
    level: f64,
) -> Result<RMatrix<f64>> {
    let fitted = &state.inner;
    let data = design(&x)?;
    let intervals = match kind {
        "credible" => fitted.credible_interval(&data, level),
        "prediction" => fitted.prediction_interval(&data, level),
        other => return Err(Error::Other(format!("unknown interval kind {other}"))),
    }
    .map_err(core_error)?;
    Ok(RMatrix::new_matrix(intervals.len(), 2, |row, col| {
        if col == 0 {
            intervals[row].lower
        } else {
            intervals[row].upper
        }
    }))
}

/// Pointwise log-likelihood, one row per draw.
#[extendr]
fn core_log_lik(
    state: ExternalPtr<FittedHandle>,
    x: RMatrix<f64>,
    y: &[f64],
) -> Result<RMatrix<f64>> {
    let draws = state
        .inner
        .log_likelihood(&design(&x)?, y)
        .map_err(core_error)?;
    Ok(draw_matrix(&draws))
}

/// sigma per kept draw; empty outside the Gaussian model.
#[extendr]
fn core_sigma(state: ExternalPtr<FittedHandle>) -> Result<Vec<f64>> {
    Ok(state.inner.sigma())
}

/// The per-draw ensemble summaries and the covariate inclusion shares.
#[extendr]
fn core_diagnostics(state: ExternalPtr<FittedHandle>) -> Result<List> {
    let fitted = &state.inner;
    Ok(list!(
        cell_count = fitted.cell_counts(),
        dimension_count = fitted.dimension_counts(),
        inclusion = fitted.variable_inclusion_proportions()
    ))
}

/// A live sampler over the core's Gibbs loop, held behind an external
/// pointer. `finish` consumes the state; every later call errors.
struct SamplerHandle {
    inner: RefCell<Option<thiessen::Sampler>>,
    data: thiessen::Data,
    y: RefCell<Vec<f64>>,
}

fn finished() -> Error {
    Error::Other("the sampler is finished".to_string())
}

/// Construct a sampler on the seed the core derives for chain `chain`, so
/// driving the configured schedule by hand reproduces that chain of a fit
/// bit for bit.
#[extendr]
fn core_sampler_new(
    config_json: &str,
    x: RMatrix<f64>,
    y: &[f64],
    seed_value: f64,
    chain: i32,
) -> Result<ExternalPtr<SamplerHandle>> {
    let config = config(config_json)?;
    let data = design(&x)?;
    let chain = thiessen::chain_seed(seed(seed_value)?, chain.max(0) as usize);
    let inner = thiessen::Sampler::new(&config, &data, y, chain).map_err(core_error)?;
    Ok(ExternalPtr::new(SamplerHandle {
        inner: RefCell::new(Some(inner)),
        data,
        y: RefCell::new(y.to_vec()),
    }))
}

/// Run `n` sweeps of the Gibbs loop.
#[extendr]
fn core_sampler_step(sampler: ExternalPtr<SamplerHandle>, n: i32) -> Result<()> {
    let mut inner = sampler.inner.borrow_mut();
    let live = inner.as_mut().ok_or_else(finished)?;
    for _ in 0..n.max(0) {
        live.step();
    }
    Ok(())
}

/// Record the current state as a posterior draw.
#[extendr]
fn core_sampler_keep(sampler: ExternalPtr<SamplerHandle>) -> Result<()> {
    sampler
        .inner
        .borrow_mut()
        .as_mut()
        .ok_or_else(finished)?
        .keep();
    Ok(())
}

/// Number of draws kept so far.
#[extendr]
fn core_sampler_n_kept(sampler: ExternalPtr<SamplerHandle>) -> Result<i32> {
    Ok(sampler
        .inner
        .borrow()
        .as_ref()
        .ok_or_else(finished)?
        .n_kept() as i32)
}

/// Replace the response on the caller's scale.
#[extendr]
fn core_sampler_set_response(sampler: ExternalPtr<SamplerHandle>, y: &[f64]) -> Result<()> {
    sampler
        .inner
        .borrow_mut()
        .as_mut()
        .ok_or_else(finished)?
        .set_response(y)
        .map_err(core_error)?;
    *sampler.y.borrow_mut() = y.to_vec();
    Ok(())
}

/// The current mean function at the training rows, caller scale.
#[extendr]
fn core_sampler_fitted_values(sampler: ExternalPtr<SamplerHandle>) -> Result<Vec<f64>> {
    Ok(sampler
        .inner
        .borrow()
        .as_ref()
        .ok_or_else(finished)?
        .fitted_values())
}

/// The current variance of y given f at each training row, caller scale.
#[extendr]
fn core_sampler_noise_variances(sampler: ExternalPtr<SamplerHandle>) -> Result<Vec<f64>> {
    Ok(sampler
        .inner
        .borrow()
        .as_ref()
        .ok_or_else(finished)?
        .noise_variances())
}

/// The fitted model from the kept draws of every sampler, their chains
/// pooled. Consumes the samplers.
#[extendr]
fn core_finish(samplers: List) -> Result<List> {
    let handles = samplers
        .values()
        .map(ExternalPtr::<SamplerHandle>::try_from)
        .collect::<Result<Vec<_>>>()?;
    let first = handles
        .first()
        .ok_or_else(|| Error::Other("a fit needs at least one chain".to_string()))?;
    let mut fits = Vec::with_capacity(handles.len());
    for handle in &handles {
        let live = handle.inner.borrow_mut().take().ok_or_else(finished)?;
        fits.push(live.finish().map_err(core_error)?);
    }
    let y = first.y.borrow();
    let fitted = thiessen::Fitted::pool(&fits, &first.data, &y).map_err(core_error)?;
    let fitted_values = fitted.predict(&first.data).map_err(core_error)?;
    let warnings: Vec<String> = fitted.warnings().iter().map(ToString::to_string).collect();
    Ok(list!(
        config = serde_json::to_string(fitted.config()).map_err(json_error)?,
        model = fitted.model_name().to_string(),
        n_chains = fits.len() as i32,
        n_draws = fitted.n_draws() as i32,
        in_sample_rmse = fitted.in_sample_rmse(),
        warnings = warnings,
        fitted_values = fitted_values,
        state = ExternalPtr::new(FittedHandle { inner: fitted })
    ))
}

extendr_module! {
    mod thiessen;
    fn core_version;
    fn core_experimental;
    fn core_defaults;
    fn core_validate;
    fn core_state_is_live;
    fn core_state_payload;
    fn core_state_restore;
    fn core_predict;
    fn core_predict_draws;
    fn core_interval;
    fn core_sigma;
    fn core_log_lik;
    fn core_diagnostics;
    fn core_sampler_new;
    fn core_sampler_step;
    fn core_sampler_keep;
    fn core_sampler_n_kept;
    fn core_sampler_set_response;
    fn core_sampler_fitted_values;
    fn core_sampler_noise_variances;
    fn core_finish;
}
