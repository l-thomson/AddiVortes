//! The interval-censored model at the fit boundary: draw-for-draw
//! agreement with the Gaussian model for exact data, the bound data
//! channel, the prediction semantics and persistence.

#![cfg(feature = "experimental")]

mod common;

use common::{fixture, SEED};
use thiessen::{fit, fit_interval_censored, Config, Error, Outcome, Sampler};

/// The Gaussian fixture as bound pairs: rows cycle through exact,
/// two-sided, censored below and censored above.
fn interval_fixture() -> (Config, thiessen::Data, Vec<f64>, Vec<f64>) {
    let (config, x, y) = fixture();
    let mut lower = Vec::with_capacity(y.len());
    let mut upper = Vec::with_capacity(y.len());
    for (i, &v) in y.iter().enumerate() {
        match i % 4 {
            0 => {
                lower.push(v);
                upper.push(v);
            }
            1 => {
                lower.push(v - 0.1);
                upper.push(v + 0.15);
            }
            2 => {
                lower.push(f64::NEG_INFINITY);
                upper.push(v);
            }
            _ => {
                lower.push(v);
                upper.push(f64::INFINITY);
            }
        }
    }
    (
        config.with_outcome(Outcome::interval_censored()),
        x,
        lower,
        upper,
    )
}

#[test]
fn exact_data_reproduces_the_gaussian_chain() {
    let (config, x, _, _) = interval_fixture();
    let (_, _, y) = fixture();
    let fitted = fit_interval_censored(&config, &x, &y, &y, SEED).unwrap();
    assert_eq!(fitted.model_name(), "interval_censored");
    let gaussian_config = Config::new().with_m(15).with_burn_in(50).with_draws(60);
    let gaussian = fit(&gaussian_config, &x, &y, SEED).unwrap();
    assert_eq!(gaussian.sigma(), fitted.sigma());
    assert_eq!(
        gaussian.predict_draws(&x).unwrap(),
        fitted.predict_draws(&x).unwrap()
    );
}

#[test]
fn censoring_changes_the_chain_and_round_trips() {
    let (config, x, lower, upper) = interval_fixture();
    let (_, _, y) = fixture();
    let exact = fit_interval_censored(&config, &x, &y, &y, SEED).unwrap();
    let fitted = fit_interval_censored(&config, &x, &lower, &upper, SEED).unwrap();
    assert_ne!(exact.sigma(), fitted.sigma());
    let json = serde_json::to_string(&fitted).unwrap();
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
    assert_eq!(back.model_name(), "interval_censored");
}

#[test]
fn the_interval_log_likelihood_carries_the_censored_terms() {
    let (config, x, lower, upper) = interval_fixture();
    let fitted = fit_interval_censored(&config, &x, &lower, &upper, SEED).unwrap();
    let (_, _, y) = fixture();
    assert!(matches!(
        fitted.log_likelihood(&x, &y),
        Err(Error::NotApplicable { ref method, .. }) if method == "log_likelihood"
    ));
    let by_draw = fitted
        .log_likelihood_interval_censored(&x, &lower, &upper)
        .unwrap();
    assert!(by_draw.iter().flatten().all(|v| v.is_finite()));
    // A censored row's term is a log interval probability, so negative.
    let row = lower
        .iter()
        .zip(&upper)
        .position(|(lo, hi)| lo < hi)
        .unwrap();
    assert!(by_draw.iter().all(|draw| draw[row] < 0.0));
    assert!(matches!(
        fitted.log_likelihood_interval_censored(&x, &upper, &lower),
        Err(Error::InvalidInterval { .. })
    ));
}

#[test]
fn the_bound_pairs_are_validated() {
    let (config, x, mut lower, mut upper) = interval_fixture();
    upper.pop();
    assert!(matches!(
        fit_interval_censored(&config, &x, &lower, &upper, SEED),
        Err(Error::BoundCountMismatch { .. })
    ));
    upper.push(f64::INFINITY);
    let held = (lower[7], upper[7]);
    (lower[7], upper[7]) = (0.4, 0.1);
    assert!(matches!(
        fit_interval_censored(&config, &x, &lower, &upper, SEED),
        Err(Error::InvalidInterval { row: 7 })
    ));
    (lower[7], upper[7]) = (f64::NAN, 0.1);
    assert!(matches!(
        fit_interval_censored(&config, &x, &lower, &upper, SEED),
        Err(Error::InvalidInterval { row: 7 })
    ));
    (lower[7], upper[7]) = (f64::NEG_INFINITY, f64::INFINITY);
    assert!(matches!(
        fit_interval_censored(&config, &x, &lower, &upper, SEED),
        Err(Error::InvalidInterval { row: 7 })
    ));
    (lower[7], upper[7]) = (f64::INFINITY, f64::INFINITY);
    assert!(matches!(
        fit_interval_censored(&config, &x, &lower, &upper, SEED),
        Err(Error::InvalidInterval { row: 7 })
    ));
    (lower[7], upper[7]) = held;
    assert!(fit_interval_censored(&config, &x, &lower, &upper, SEED).is_ok());
}

#[test]
fn the_plain_entry_points_reject_the_interval_censored_outcome() {
    let (config, x, lower, upper) = interval_fixture();
    let (_, _, y) = fixture();
    let err = fit(&config, &x, &y, SEED).unwrap_err();
    assert!(err.to_string().contains("fit_interval_censored"), "{err}");
    let gaussian = Config::new().with_m(15).with_burn_in(50).with_draws(60);
    let err = fit_interval_censored(&gaussian, &x, &lower, &upper, SEED).unwrap_err();
    assert!(
        err.to_string().contains("interval_censored outcome"),
        "{err}"
    );
}

#[test]
fn the_response_seam_replaces_the_bounds() {
    let (config, x, lower, upper) = interval_fixture();
    let mut sampler = Sampler::interval_censored(&config, &x, &lower, &upper, SEED).unwrap();
    sampler.step();
    let (_, _, y) = fixture();
    let err = sampler.set_response(&y).unwrap_err();
    assert!(
        err.to_string().contains("set_interval_censored_response"),
        "{err}"
    );
    let widened: Vec<f64> = upper.iter().map(|&v| v + 0.05).collect();
    sampler
        .set_interval_censored_response(&lower, &widened)
        .unwrap();
    sampler.step();
    assert!(sampler.sigma_sq().is_finite());
}

#[test]
fn a_variance_ensemble_composes_with_the_interval_censored_model() {
    let (config, x, lower, upper) = interval_fixture();
    let fitted = fit_interval_censored(&config.with_m_var(5), &x, &lower, &upper, SEED).unwrap();
    assert_eq!(fitted.model_name(), "interval_censored");
    assert!(fitted.sigma().is_empty());
    let variances = fitted.predict_variance(&x).unwrap();
    assert!(variances.iter().flatten().all(|v| *v > 0.0));
}

#[test]
fn the_outcome_serialises_in_snake_case_and_round_trips() {
    let (config, _, _, _) = interval_fixture();
    let json = serde_json::to_string(&config).unwrap();
    assert!(
        json.contains(r#""outcome":{"interval_censored":{"nu":6.0,"q":0.85}}"#),
        "{json}"
    );
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(config, back);
}
