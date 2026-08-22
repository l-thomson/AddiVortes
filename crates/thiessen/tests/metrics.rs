//! The experimental metrics at the fit boundary: the order-two
//! equivalence with Euclidean, the Manhattan alias, and the
//! configuration surface.

#![cfg(feature = "experimental")]

mod common;

use common::{fixture, SEED};
use thiessen::{fit, Config, GowerKind, Metric};

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

#[test]
fn cosine_changes_the_chain_and_round_trips() {
    let (config, x, y) = fixture();
    let euclidean = fit(&config.clone(), &x, &y, SEED).unwrap();
    let fitted = fit(&config.with_metric(vec![Metric::Cosine; 2]), &x, &y, SEED).unwrap();
    assert_ne!(euclidean.sigma(), fitted.sigma());
    let json = serde_json::to_string(&fitted).unwrap();
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
}

#[test]
fn the_cosine_configuration_serialises_by_name() {
    let config = Config::new().with_metric(vec![Metric::Cosine, Metric::Euclidean]);
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains(r#""cosine""#), "{json}");
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(back, config);
}

fn gower_metric() -> Vec<Metric> {
    vec![
        Metric::Gower {
            kind: GowerKind::Numeric,
        },
        Metric::Gower {
            kind: GowerKind::Categorical,
        },
    ]
}

/// The Gaussian fixture with its second column rounded to integer codes.
fn gower_fixture() -> (Config, thiessen::Data, Vec<f64>) {
    let (config, x, y) = fixture();
    let rows: Vec<Vec<f64>> = (0..x.n_rows())
        .map(|i| vec![x.row(i)[0], (x.row(i)[1] * 3.0).round()])
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    (config, thiessen::Data::from_rows(&refs).unwrap(), y)
}

#[test]
fn gower_changes_the_chain_and_round_trips() {
    let (config, x, y) = gower_fixture();
    let euclidean = fit(&config.clone(), &x, &y, SEED).unwrap();
    let fitted = fit(&config.with_metric(gower_metric()), &x, &y, SEED).unwrap();
    assert_ne!(euclidean.sigma(), fitted.sigma());
    let json = serde_json::to_string(&fitted).unwrap();
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
}

#[test]
fn gower_rejects_a_non_integer_code_at_fit() {
    let (config, x, y) = fixture();
    let err = fit(&config.with_metric(gower_metric()), &x, &y, SEED).unwrap_err();
    assert!(matches!(err, thiessen::Error::InvalidCategoryCode { .. }));
}

#[test]
fn the_gower_configuration_serialises_by_kind() {
    let config = Config::new().with_metric(gower_metric());
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains(r#"{"gower":{"kind":"numeric"}}"#), "{json}");
    assert!(
        json.contains(r#"{"gower":{"kind":"categorical"}}"#),
        "{json}"
    );
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(back, config);
}
