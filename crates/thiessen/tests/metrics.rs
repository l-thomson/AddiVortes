//! The experimental metrics at the fit boundary: the order-two
//! equivalence with Euclidean, the Manhattan alias, and the
//! configuration surface.

#![cfg(feature = "experimental")]

mod common;

use common::{fixture, SEED};
use thiessen::{fit, Config, Metric};

fn minkowski(p: f64) -> Vec<Metric> {
    vec![Metric::Minkowski { p }; 2]
}

#[test]
fn order_two_reproduces_the_euclidean_chain() {
    let (config, x, y) = fixture();
    let euclidean = fit(&config.clone(), &x, &y, SEED).unwrap();
    let fitted = fit(&config.with_metric(minkowski(2.0)), &x, &y, SEED).unwrap();
    assert_eq!(euclidean.sigma(), fitted.sigma());
    assert_eq!(
        euclidean.predict_draws(&x).unwrap(),
        fitted.predict_draws(&x).unwrap()
    );
}

#[test]
fn manhattan_is_minkowski_of_order_one() {
    let (config, x, y) = fixture();
    let named = fit(
        &config.clone().with_metric(vec![Metric::Manhattan; 2]),
        &x,
        &y,
        SEED,
    )
    .unwrap();
    let ordered = fit(&config.with_metric(minkowski(1.0)), &x, &y, SEED).unwrap();
    assert_eq!(named.sigma(), ordered.sigma());
    assert_eq!(
        named.predict_draws(&x).unwrap(),
        ordered.predict_draws(&x).unwrap()
    );
}

#[test]
fn order_one_changes_the_chain() {
    let (config, x, y) = fixture();
    let euclidean = fit(&config.clone(), &x, &y, SEED).unwrap();
    let fitted = fit(&config.with_metric(minkowski(1.0)), &x, &y, SEED).unwrap();
    assert_ne!(euclidean.sigma(), fitted.sigma());
}

#[test]
fn the_configuration_round_trips() {
    let config = Config::new().with_metric(vec![Metric::Minkowski { p: 1.5 }, Metric::Manhattan]);
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains(r#"{"minkowski":{"p":1.5}}"#), "{json}");
    assert!(json.contains(r#""manhattan""#), "{json}");
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(back, config);
}

#[test]
fn a_fit_round_trips_through_its_saved_state() {
    let (config, x, y) = fixture();
    let fitted = fit(&config.with_metric(minkowski(1.0)), &x, &y, SEED).unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
}
