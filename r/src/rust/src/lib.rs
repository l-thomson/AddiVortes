//! extendr bindings over the `thiessen` core: the configuration as JSON,
//! the design as an R matrix, and the fitted model as the JSON of its serde
//! representation.
//!
//! The configuration crosses the boundary as JSON so that the core's serde
//! representation stays the single definition of the field names, their
//! defaults and their validation. The fitted model crosses the same way, so
//! an R fit is a plain R object that `saveRDS` writes and a later session
//! reads.

use extendr_api::prelude::*;
use extendr_api::Result;

fn core_error(error: thiessen::Error) -> Error {
    Error::Other(error.to_string())
}

fn json_error(error: serde_json::Error) -> Error {
    Error::Other(error.to_string())
}

fn config(json: &str) -> Result<thiessen::Config> {
    serde_json::from_str(json).map_err(json_error)
}

fn state(json: &str) -> Result<thiessen::Fitted> {
    serde_json::from_str(json).map_err(json_error)
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

/// Fit the model of the configuration and return the fit and the quantities
/// the print, summary, fitted and residuals methods report.
#[extendr]
fn core_fit(config_json: &str, x: RMatrix<f64>, y: &[f64], seed_value: f64) -> Result<List> {
    let config = config(config_json)?;
    let data = design(&x)?;
    let fitted = thiessen::fit(&config, &data, y, seed(seed_value)?).map_err(core_error)?;
    let fitted_values = fitted.predict(&data).map_err(core_error)?;
    let warnings: Vec<String> = fitted.warnings().iter().map(ToString::to_string).collect();
    Ok(list!(
        state = serde_json::to_string(&fitted).map_err(json_error)?,
        config = serde_json::to_string(fitted.config()).map_err(json_error)?,
        model = fitted.model().to_string(),
        n_draws = fitted.n_draws() as i32,
        in_sample_rmse = fitted.in_sample_rmse(),
        warnings = warnings,
        fitted_values = fitted_values
    ))
}

/// The posterior mean at each row of `x`.
#[extendr]
fn core_predict(state_json: &str, x: RMatrix<f64>) -> Result<Vec<f64>> {
    state(state_json)?.predict(&design(&x)?).map_err(core_error)
}

/// Per-draw predictions: `kind` selects the quantity of `predict`, the mean
/// function, or the variance of y given f.
#[extendr]
fn core_predict_draws(state_json: &str, x: RMatrix<f64>, kind: &str) -> Result<RMatrix<f64>> {
    let fitted = state(state_json)?;
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
    state_json: &str,
    x: RMatrix<f64>,
    kind: &str,
    level: f64,
) -> Result<RMatrix<f64>> {
    let fitted = state(state_json)?;
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
fn core_log_lik(state_json: &str, x: RMatrix<f64>, y: &[f64]) -> Result<RMatrix<f64>> {
    let draws = state(state_json)?
        .log_likelihood(&design(&x)?, y)
        .map_err(core_error)?;
    Ok(draw_matrix(&draws))
}

/// sigma per kept draw; empty outside the Gaussian model.
#[extendr]
fn core_sigma(state_json: &str) -> Result<Vec<f64>> {
    Ok(state(state_json)?.sigma())
}

/// The per-draw ensemble summaries and the covariate inclusion shares.
#[extendr]
fn core_diagnostics(state_json: &str) -> Result<List> {
    let fitted = state(state_json)?;
    Ok(list!(
        cell_count = fitted.cell_counts(),
        dimension_count = fitted.dimension_counts(),
        inclusion = fitted.variable_inclusion_proportions()
    ))
}

extendr_module! {
    mod thiessen;
    fn core_version;
    fn core_defaults;
    fn core_validate;
    fn core_fit;
    fn core_predict;
    fn core_predict_draws;
    fn core_interval;
    fn core_sigma;
    fn core_log_lik;
    fn core_diagnostics;
}
