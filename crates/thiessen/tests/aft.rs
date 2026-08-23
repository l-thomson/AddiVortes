//! The AFT model at the fit boundary: draw-for-draw agreement with the
//! Gaussian model on log times for all-event data, the survival data
//! channel, the prediction semantics and persistence.

#![cfg(feature = "experimental")]

mod common;

use common::{fixture, SEED};
use thiessen::{fit, fit_aft, Config, Error, Outcome, Sampler};

/// The Gaussian fixture as survival data: times exp(y), a third of the
/// rows right-censored at their own times.
fn aft_fixture() -> (Config, thiessen::Data, Vec<f64>, Vec<bool>) {
    let (config, x, y) = fixture();
    let times: Vec<f64> = y.iter().map(|&v| v.exp()).collect();
    let events: Vec<bool> = (0..times.len()).map(|i| i % 3 != 0).collect();
    (config.with_outcome(Outcome::aft()), x, times, events)
}

#[test]
fn all_event_data_reproduces_the_gaussian_chain_on_log_times() {
    let (config, x, times, _) = aft_fixture();
    let events = vec![true; times.len()];
    let aft = fit_aft(&config, &x, &times, &events, SEED).unwrap();
    assert_eq!(aft.model_name(), "aft");
    // The engine's response is the libm logarithm of the times.
    let log_times: Vec<f64> = times.iter().map(|&t| libm::log(t)).collect();
    let gaussian_config = Config::new().with_m(15).with_burn_in(50).with_draws(60);
    let gaussian = fit(&gaussian_config, &x, &log_times, SEED).unwrap();
    assert_eq!(gaussian.sigma(), aft.sigma());
    assert_eq!(
        gaussian.predict_draws(&x).unwrap(),
        aft.predict_draws(&x).unwrap()
    );
}

#[test]
fn censoring_changes_the_chain_and_round_trips() {
    let (config, x, times, events) = aft_fixture();
    let all_events = fit_aft(&config, &x, &times, &vec![true; times.len()], SEED).unwrap();
    let fitted = fit_aft(&config, &x, &times, &events, SEED).unwrap();
    assert_ne!(all_events.sigma(), fitted.sigma());
    let json = serde_json::to_string(&fitted).unwrap();
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
    assert_eq!(back.model_name(), "aft");
}

#[test]
fn the_survival_log_likelihood_carries_the_censored_terms() {
    let (config, x, times, events) = aft_fixture();
    let fitted = fit_aft(&config, &x, &times, &events, SEED).unwrap();
    let log_times: Vec<f64> = times.iter().map(|&t| t.ln()).collect();
    assert!(matches!(
        fitted.log_likelihood(&x, &log_times),
        Err(Error::NotApplicable { ref method, .. }) if method == "log_likelihood"
    ));
    let by_draw = fitted.log_likelihood_survival(&x, &times, &events).unwrap();
    assert!(by_draw.iter().flatten().all(|v| v.is_finite()));
    // A censored row's term is a log survival probability, so negative.
    let row = events.iter().position(|&event| !event).unwrap();
    assert!(by_draw.iter().all(|draw| draw[row] < 0.0));
}

#[test]
fn the_survival_data_are_validated() {
    let (config, x, mut times, mut events) = aft_fixture();
    events.pop();
    assert!(matches!(
        fit_aft(&config, &x, &times, &events, SEED),
        Err(Error::EventCountMismatch { .. })
    ));
    events.push(true);
    times[7] = 0.0;
    assert!(matches!(
        fit_aft(&config, &x, &times, &events, SEED),
        Err(Error::InvalidSurvivalTime { row: 7 })
    ));
    times[7] = f64::NAN;
    assert!(matches!(
        fit_aft(&config, &x, &times, &events, SEED),
        Err(Error::InvalidSurvivalTime { row: 7 })
    ));
}

#[test]
fn the_plain_entry_points_reject_the_aft_outcome() {
    let (config, x, times, events) = aft_fixture();
    let log_times: Vec<f64> = times.iter().map(|&t| t.ln()).collect();
    let err = fit(&config, &x, &log_times, SEED).unwrap_err();
    assert!(err.to_string().contains("fit_aft"), "{err}");
    let gaussian = Config::new().with_m(15).with_burn_in(50).with_draws(60);
    let err = fit_aft(&gaussian, &x, &times, &events, SEED).unwrap_err();
    assert!(err.to_string().contains("aft outcome"), "{err}");
}

#[test]
fn the_response_seam_replaces_times_and_events() {
    let (config, x, times, events) = aft_fixture();
    let mut sampler = Sampler::aft(&config, &x, &times, &events, SEED).unwrap();
    sampler.step();
    let err = sampler.set_response(&times).unwrap_err();
    assert!(err.to_string().contains("set_aft_response"), "{err}");
    let flipped: Vec<bool> = events.iter().map(|&event| !event).collect();
    sampler.set_aft_response(&times, &flipped).unwrap();
    sampler.step();
    assert!(sampler.sigma_sq().is_finite());
}

#[test]
fn a_variance_ensemble_composes_with_the_aft_model() {
    let (config, x, times, events) = aft_fixture();
    let fitted = fit_aft(&config.with_m_var(5), &x, &times, &events, SEED).unwrap();
    assert_eq!(fitted.model_name(), "aft");
    assert!(fitted.sigma().is_empty());
    let variances = fitted.predict_variance(&x).unwrap();
    assert!(variances.iter().flatten().all(|v| *v > 0.0));
}

#[test]
fn the_outcome_serialises_in_snake_case_and_round_trips() {
    let (config, _, _, _) = aft_fixture();
    let json = serde_json::to_string(&config).unwrap();
    assert!(
        json.contains(r#""outcome":{"aft":{"nu":6.0,"q":0.85}}"#),
        "{json}"
    );
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(config, back);
}
