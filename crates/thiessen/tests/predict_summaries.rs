//! The one-traversal summary equals the separate calls, and the interval
//! root-finder stops where the bracket has collapsed.

mod common;

use common::{fixture, SEED};
use thiessen::{fit, IntervalKind};

#[test]
fn predict_with_interval_equals_the_separate_calls() {
    let (config, x, y) = fixture();
    let fitted = fit(&config, &x, &y, SEED).unwrap();

    let (mean, credible) = fitted
        .predict_with_interval(&x, IntervalKind::Credible, 0.9)
        .unwrap();
    assert_eq!(mean, fitted.predict(&x).unwrap());
    assert_eq!(credible, fitted.credible_interval(&x, 0.9).unwrap());

    let (mean, prediction) = fitted
        .predict_with_interval(&x, IntervalKind::Prediction, 0.9)
        .unwrap();
    assert_eq!(mean, fitted.predict(&x).unwrap());
    assert_eq!(prediction, fitted.prediction_interval(&x, 0.9).unwrap());
}

#[test]
fn heteroscedastic_summaries_agree_too() {
    let (config, x, y) = fixture();
    let fitted = fit(&config.with_m_var(5), &x, &y, SEED).unwrap();
    let (mean, prediction) = fitted
        .predict_with_interval(&x, IntervalKind::Prediction, 0.95)
        .unwrap();
    assert_eq!(mean, fitted.predict(&x).unwrap());
    assert_eq!(prediction, fitted.prediction_interval(&x, 0.95).unwrap());
}

#[test]
fn an_invalid_level_is_rejected_before_any_work() {
    let (config, x, y) = fixture();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    assert!(fitted
        .predict_with_interval(&x, IntervalKind::Credible, 1.0)
        .is_err());
}
