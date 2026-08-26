//! Python bindings over [`thiessen`]: the configuration as JSON, the design
//! and response as numpy arrays, and the fitted model as an opaque handle.
//!
//! The configuration crosses the boundary as JSON so that the core's serde
//! representation stays the single definition of the field names, their
//! defaults and their validation, including the error naming the
//! `experimental` feature.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

create_exception!(
    _native,
    ThiessenError,
    PyValueError,
    "Invalid configuration, data or fitted model."
);

fn core_error(error: thiessen::Error) -> PyErr {
    ThiessenError::new_err(error.to_string())
}

fn config(json: &str) -> PyResult<thiessen::Config> {
    serde_json::from_str(json).map_err(|e| ThiessenError::new_err(e.to_string()))
}

fn design(x: &PyReadonlyArray2<'_, f64>) -> PyResult<thiessen::Data> {
    let view = x.as_array();
    let (n_rows, n_cols) = view.dim();
    // `ArrayView::iter` visits in logical order, so the values are row-major
    // whatever the array's memory layout.
    let values: Vec<f64> = view.iter().copied().collect();
    thiessen::Data::new(values, n_rows, n_cols).map_err(core_error)
}

fn response(y: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    y.as_array().iter().copied().collect()
}

fn matrix<'py>(py: Python<'py>, rows: Vec<Vec<f64>>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    PyArray2::from_vec2(py, &rows).map_err(|e| ThiessenError::new_err(e.to_string()))
}

/// A fitted model.
#[pyclass(module = "thiessen._native", name = "Fitted")]
pub struct Fitted {
    inner: thiessen::Fitted,
}

#[pymethods]
impl Fitted {
    /// The number of threads the predictions run on.
    #[getter]
    fn threads(&self) -> usize {
        self.inner.threads()
    }

    /// Set the number of threads the predictions run on.
    fn set_threads(&mut self, n_threads: usize) {
        self.inner.set_threads(n_threads);
    }

    /// The posterior mean at each row of `x`.
    fn predict<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = self.inner.predict(&design(&x)?).map_err(core_error)?;
        Ok(PyArray1::from_vec(py, values))
    }

    /// The quantity of `predict` for every kept draw, draw-major.
    fn predict_draws<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let draws = self.inner.predict_draws(&design(&x)?).map_err(core_error)?;
        matrix(py, draws)
    }

    /// The mean function for every kept draw, draw-major.
    fn predict_latent<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let draws = self
            .inner
            .predict_latent(&design(&x)?)
            .map_err(core_error)?;
        matrix(py, draws)
    }

    /// The variance of y given f for every kept draw, draw-major.
    fn predict_variance<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let draws = self
            .inner
            .predict_variance(&design(&x)?)
            .map_err(core_error)?;
        matrix(py, draws)
    }

    /// Posterior quantiles of the quantity of `predict`, `n_rows` by
    /// `probs.len()`.
    fn predict_quantiles<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        probs: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let data = design(&x)?;
        let flat = self
            .inner
            .predict_quantiles(&data, &probs)
            .map_err(core_error)?;
        let rows: Vec<Vec<f64>> = flat
            .chunks(probs.len().max(1))
            .map(<[f64]>::to_vec)
            .collect();
        matrix(py, rows)
    }

    /// Central credible interval at `level`, `n_rows` by 2.
    fn credible_interval<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        level: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let intervals = self
            .inner
            .credible_interval(&design(&x)?, level)
            .map_err(core_error)?;
        matrix(
            py,
            intervals.iter().map(|i| vec![i.lower, i.upper]).collect(),
        )
    }

    /// Central posterior predictive interval at `level`, `n_rows` by 2.
    fn prediction_interval<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        level: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let intervals = self
            .inner
            .prediction_interval(&design(&x)?, level)
            .map_err(core_error)?;
        matrix(
            py,
            intervals.iter().map(|i| vec![i.lower, i.upper]).collect(),
        )
    }

    /// Pointwise log-likelihood per draw, draw-major.
    fn log_likelihood<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let draws = self
            .inner
            .log_likelihood(&design(&x)?, &response(&y))
            .map_err(core_error)?;
        matrix(py, draws)
    }

    /// sigma per kept draw; empty outside the Gaussian model.
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.sigma())
    }

    /// Mean cells per mean tessellation, per kept draw.
    fn cell_counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.cell_counts())
    }

    /// Mean active covariates per mean tessellation, per kept draw.
    fn dimension_counts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.dimension_counts())
    }

    /// Share of active dimensions falling on each covariate.
    fn variable_inclusion_proportions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.inner.variable_inclusion_proportions())
    }

    /// Number of kept draws.
    #[getter]
    fn n_draws(&self) -> usize {
        self.inner.n_draws()
    }

    /// The observation model.
    #[getter]
    fn model(&self) -> String {
        self.inner.model_name().to_string()
    }

    /// Root mean squared error of the posterior mean on the training rows.
    #[getter]
    fn in_sample_rmse(&self) -> f64 {
        self.inner.in_sample_rmse()
    }

    /// The fit-time warnings, one message each.
    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner
            .warnings()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// The resolved configuration as JSON.
    #[getter]
    fn config(&self) -> PyResult<String> {
        serde_json::to_string(self.inner.config())
            .map_err(|e| ThiessenError::new_err(e.to_string()))
    }

    /// The fitted model as JSON.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|e| ThiessenError::new_err(e.to_string()))
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let loader = py.import("thiessen._native")?.getattr("fitted_from_json")?;
        Ok((loader, (self.to_json()?,)))
    }
}

/// A live sampler over the core's Gibbs loop.
///
/// Owns its RNG, seeded at construction with the chain-0 seed of `fit`, so
/// driving the configured schedule by hand reproduces `fit` bit for bit.
/// `finish` consumes the state; every later call fails.
#[pyclass(module = "thiessen._native", name = "Sampler")]
pub struct Sampler {
    inner: Option<thiessen::Sampler>,
    data: thiessen::Data,
    y: Vec<f64>,
}

impl Sampler {
    fn live(&self) -> PyResult<&thiessen::Sampler> {
        self.inner
            .as_ref()
            .ok_or_else(|| ThiessenError::new_err("the sampler is finished"))
    }

    fn live_mut(&mut self) -> PyResult<&mut thiessen::Sampler> {
        self.inner
            .as_mut()
            .ok_or_else(|| ThiessenError::new_err("the sampler is finished"))
    }
}

#[pymethods]
impl Sampler {
    #[new]
    fn new(
        config_json: &str,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray1<'_, f64>,
        seed: u64,
    ) -> PyResult<Self> {
        let config = config(config_json)?;
        let data = design(&x)?;
        let y = response(&y);
        let chain = thiessen::chain_seed(seed, 0);
        let inner = thiessen::Sampler::new(&config, &data, &y, chain).map_err(core_error)?;
        Ok(Self {
            inner: Some(inner),
            data,
            y,
        })
    }

    /// Run `n` sweeps of the Gibbs loop.
    fn step(&mut self, n: usize) -> PyResult<()> {
        let sampler = self.live_mut()?;
        for _ in 0..n {
            sampler.step();
        }
        Ok(())
    }

    /// Record the current state as a posterior draw.
    fn keep(&mut self) -> PyResult<()> {
        self.live_mut()?.keep();
        Ok(())
    }

    /// Replace the response on the caller's scale.
    fn set_response(&mut self, y: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        let values = response(&y);
        self.live_mut()?.set_response(&values).map_err(core_error)?;
        self.y = values;
        Ok(())
    }

    /// The current mean function at the training rows, caller scale.
    fn fitted_values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray1::from_vec(py, self.live()?.fitted_values()))
    }

    /// The variance of y given f at each training row, caller scale.
    fn noise_variances<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(PyArray1::from_vec(py, self.live()?.noise_variances()))
    }

    /// Number of draws kept so far.
    #[getter]
    fn n_kept(&self) -> PyResult<usize> {
        Ok(self.live()?.n_kept())
    }

    /// The configuration, with omega and the probit offset resolved.
    #[getter]
    fn config(&self) -> PyResult<String> {
        serde_json::to_string(self.live()?.config())
            .map_err(|e| ThiessenError::new_err(e.to_string()))
    }

    /// The fitted model from the kept draws, pooled as a one-chain fit.
    fn finish(&mut self) -> PyResult<Fitted> {
        let sampler = self
            .inner
            .take()
            .ok_or_else(|| ThiessenError::new_err("the sampler is finished"))?;
        let (inner, _) = thiessen::Fitted::pool_samplers(vec![sampler], &self.data, &self.y)
            .map_err(core_error)?;
        Ok(Fitted { inner })
    }
}

/// Fit the model of the configuration to `x` and `y` under `seed`, running
/// `n_chains` chains seeded by [`thiessen::chain_seed`] over at most
/// `n_threads` threads and pooling their draws. The GIL is released for
/// the fit.
#[pyfunction]
fn fit(
    py: Python<'_>,
    config_json: &str,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    seed: u64,
    n_chains: usize,
    n_threads: usize,
) -> PyResult<Fitted> {
    let config = config(config_json)?;
    let data = design(&x)?;
    let y = response(&y);
    let (inner, _) = py
        .detach(|| {
            thiessen::fit_chains_with_threads(
                &config,
                &data,
                &y,
                seed,
                n_chains.max(1),
                n_threads.max(1),
            )
        })
        .map_err(core_error)?;
    Ok(Fitted { inner })
}

/// Load a fitted model from the JSON of `Fitted.to_json`, predicting on
/// `n_threads` threads.
#[pyfunction]
#[pyo3(signature = (json, n_threads = 1))]
fn fitted_from_json(json: &str, n_threads: usize) -> PyResult<Fitted> {
    let mut inner: thiessen::Fitted =
        serde_json::from_str(json).map_err(|e| ThiessenError::new_err(e.to_string()))?;
    inner.set_threads(n_threads);
    Ok(Fitted { inner })
}

/// Validate a configuration without data.
#[pyfunction]
fn validate_config(config_json: &str) -> PyResult<()> {
    config(config_json)?.validate().map_err(core_error)
}

/// The core's default configuration as JSON.
#[pyfunction]
fn default_config() -> PyResult<String> {
    serde_json::to_string(&thiessen::Config::new())
        .map_err(|e| ThiessenError::new_err(e.to_string()))
}

/// The default parameters of each outcome family, as JSON.
#[pyfunction]
fn outcome_defaults() -> PyResult<String> {
    let families = [thiessen::Outcome::gaussian(), thiessen::Outcome::probit()];
    serde_json::to_string(&families).map_err(|e| ThiessenError::new_err(e.to_string()))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CORE_VERSION", thiessen::VERSION)?;
    m.add("EXPERIMENTAL", cfg!(feature = "experimental"))?;
    m.add("ThiessenError", m.py().get_type::<ThiessenError>())?;
    m.add_class::<Fitted>()?;
    m.add_class::<Sampler>()?;
    m.add_function(wrap_pyfunction!(fit, m)?)?;
    m.add_function(wrap_pyfunction!(fitted_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(validate_config, m)?)?;
    m.add_function(wrap_pyfunction!(default_config, m)?)?;
    m.add_function(wrap_pyfunction!(outcome_defaults, m)?)?;
    Ok(())
}
