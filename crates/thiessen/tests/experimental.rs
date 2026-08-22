//! A build without the `experimental` feature rejects configurations that
//! name an experimental option.

#![cfg(not(feature = "experimental"))]

use thiessen::Config;

#[test]
fn unknown_config_field_is_rejected() {
    let json = r#"{"mean_params": {"geometry": {"membership": "soft"}}}"#;
    let err = serde_json::from_str::<Config>(json).unwrap_err();
    assert!(err.to_string().contains("membership"), "{err}");
}

#[test]
fn unknown_outcome_is_rejected() {
    let err = serde_json::from_str::<Config>(r#"{"outcome": {"robust": {}}}"#).unwrap_err();
    assert!(err.to_string().contains("robust"), "{err}");
}

#[test]
fn minkowski_metric_is_rejected() {
    let json = r#"{"mean_params": {"geometry": {"metric": [{"minkowski": {"p": 1.5}}]}}}"#;
    let err = serde_json::from_str::<Config>(json).unwrap_err();
    assert!(err.to_string().contains("minkowski"), "{err}");
}

#[test]
fn manhattan_metric_is_rejected() {
    let json = r#"{"mean_params": {"geometry": {"metric": ["manhattan"]}}}"#;
    let err = serde_json::from_str::<Config>(json).unwrap_err();
    assert!(err.to_string().contains("manhattan"), "{err}");
}

#[test]
fn cosine_metric_is_rejected() {
    let json = r#"{"mean_params": {"geometry": {"metric": ["cosine"]}}}"#;
    let err = serde_json::from_str::<Config>(json).unwrap_err();
    assert!(err.to_string().contains("cosine"), "{err}");
}
