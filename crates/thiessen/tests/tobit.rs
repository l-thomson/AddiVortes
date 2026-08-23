//! The tobit model at the fit boundary: draw-for-draw agreement with the
//! Gaussian model on uncensored data, the configuration surface, the
//! censored prediction semantics and persistence.

#![cfg(feature = "experimental")]

mod common;

use common::{fixture, SEED};
use thiessen::{fit, Config, Data, Error, Outcome};

/// The Gaussian fixture censored at fixed limits: values beyond a limit
/// clamped to it, several rows on each side.
fn tobit_fixture() -> (Config, Data, Vec<f64>, f64, f64) {
    let (config, x, y) = fixture();
    let (lower, upper) = (0.3, 1.1);
    let censored: Vec<f64> = y.iter().map(|&v| v.clamp(lower, upper)).collect();
    assert!(censored.iter().filter(|&&v| v == lower).count() >= 3);
    assert!(censored.iter().filter(|&&v| v == upper).count() >= 3);
    let config = config.with_outcome(Outcome::tobit(Some(lower), Some(upper)));
    (config, x, censored, lower, upper)
}

#[test]
fn uncensored_data_reproduces_the_gaussian_chain() {
    let (config, x, y) = fixture();
    let gaussian = fit(&config.clone(), &x, &y, SEED).unwrap();
    let limits = Outcome::tobit(Some(-100.0), Some(100.0));
    let tobit = fit(&config.with_outcome(limits), &x, &y, SEED).unwrap();
    assert_eq!(tobit.model_name(), "tobit");
    assert_eq!(gaussian.sigma(), tobit.sigma());
    assert_eq!(
        gaussian.predict_draws(&x).unwrap(),
        tobit.predict_draws(&x).unwrap()
    );
    assert_eq!(gaussian.in_sample_rmse(), tobit.in_sample_rmse());
}

#[test]
fn censored_data_changes_the_chain_and_round_trips() {
    let (config, x, y, _, _) = tobit_fixture();
    let gaussian = fit(
        &Config::new().with_m(15).with_burn_in(50).with_draws(60),
        &x,
        &y,
        SEED,
    )
    .unwrap();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    assert_ne!(gaussian.sigma(), fitted.sigma());
    let json = serde_json::to_string(&fitted).unwrap();
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
    assert_eq!(back.model_name(), "tobit");
}

#[test]
fn the_prediction_interval_is_clamped_to_the_limits() {
    let (config, x, y, lower, upper) = tobit_fixture();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    let intervals = fitted.prediction_interval(&x, 0.99).unwrap();
    for interval in &intervals {
        assert!(interval.lower >= lower && interval.upper <= upper);
        assert!(interval.lower <= interval.upper);
    }
    // The latent mean itself is not clamped.
    assert!(fitted.predict(&x).unwrap().iter().all(|v| v.is_finite()));
}

#[test]
fn the_log_likelihood_carries_the_censored_terms() {
    let (config, x, y, lower, _) = tobit_fixture();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    let by_draw = fitted.log_likelihood(&x, &y).unwrap();
    assert!(by_draw.iter().flatten().all(|v| v.is_finite()));
    // A censored row's term is a log probability, so it is negative;
    // moving the response beyond a limit is rejected.
    let row = y.iter().position(|&v| v == lower).unwrap();
    assert!(by_draw.iter().all(|draw| draw[row] < 0.0));
    let mut beyond = y.clone();
    beyond[row] = lower - 0.5;
    assert!(matches!(
        fitted.log_likelihood(&x, &beyond),
        Err(Error::ResponseBeyondLimit { row: r }) if r == row
    ));
}

#[test]
fn a_variance_ensemble_composes_with_the_tobit_model() {
    let (config, x, y, _, _) = tobit_fixture();
    let fitted = fit(&config.with_m_var(5), &x, &y, SEED).unwrap();
    assert_eq!(fitted.model_name(), "tobit");
    assert!(fitted.sigma().is_empty());
    let variances = fitted.predict_variance(&x).unwrap();
    assert!(variances.iter().flatten().all(|v| *v > 0.0));
}

#[test]
fn a_response_beyond_a_limit_is_rejected() {
    let (config, x, mut y, lower, _) = tobit_fixture();
    y[5] = lower - 1.0;
    assert!(matches!(
        fit(&config, &x, &y, SEED),
        Err(Error::ResponseBeyondLimit { row: 5 })
    ));
}

#[test]
fn the_limits_are_validated() {
    let (config, x, y) = fixture();
    let no_limits = config.clone().with_outcome(Outcome::tobit(None, None));
    let err = fit(&no_limits, &x, &y, SEED).unwrap_err();
    assert!(err.to_string().contains("censoring limit"), "{err}");
    let crossed = config
        .clone()
        .with_outcome(Outcome::tobit(Some(1.0), Some(0.5)));
    let err = fit(&crossed, &x, &y, SEED).unwrap_err();
    assert!(err.to_string().contains("below upper"), "{err}");
    let non_finite = config.with_outcome(Outcome::tobit(Some(f64::NAN), None));
    let err = fit(&non_finite, &x, &y, SEED).unwrap_err();
    assert!(err.to_string().contains("finite"), "{err}");
}

#[test]
fn the_outcome_serialises_in_snake_case_and_round_trips() {
    let (config, _, _, _, _) = tobit_fixture();
    let json = serde_json::to_string(&config).unwrap();
    assert!(
        json.contains(r#""outcome":{"tobit":{"lower":0.3,"upper":1.1,"nu":6.0,"q":0.85}}"#),
        "{json}"
    );
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(config, back);
}
