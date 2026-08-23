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
fn the_tobit_outcome_is_rejected_naming_the_feature() {
    let json = r#"{"outcome": {"tobit": {"lower": 0.0}}}"#;
    let err = serde_json::from_str::<Config>(json).unwrap_err();
    assert!(err.to_string().contains("experimental"), "{err}");
    assert!(err.to_string().contains("tobit"), "{err}");
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

#[test]
fn gower_metric_is_rejected() {
    let json = r#"{"mean_params": {"geometry": {"metric": [{"gower": {"kind": "numeric"}}]}}}"#;
    let err = serde_json::from_str::<Config>(json).unwrap_err();
    assert!(err.to_string().contains("gower"), "{err}");
}

#[test]
fn mahalanobis_metric_is_rejected() {
    let json = r#"{"mean_params": {"geometry": {"metric": ["mahalanobis"]}}}"#;
    let err = serde_json::from_str::<Config>(json).unwrap_err();
    assert!(err.to_string().contains("mahalanobis"), "{err}");
}

#[test]
fn the_precision_field_is_rejected() {
    let json = r#"{"mean_params": {"geometry": {"precision": [1.0, 0.0, 0.0, 1.0]}}}"#;
    let err = serde_json::from_str::<Config>(json).unwrap_err();
    assert!(err.to_string().contains("precision"), "{err}");
}

#[test]
fn the_inclusion_field_is_rejected() {
    let json = r#"{"mean_params": {"structure": {"inclusion": "uniform"}}}"#;
    let err = serde_json::from_str::<Config>(json).unwrap_err();
    assert!(err.to_string().contains("inclusion"), "{err}");
}

#[test]
fn a_saved_bandwidth_is_rejected() {
    let json = r#"{"centres":[0.1],"dims":[0],"mus":[1.0],"tau":0.2}"#;
    let err = serde_json::from_str::<thiessen::Tessellation>(json).unwrap_err();
    assert!(err.to_string().contains("experimental"), "{err}");
}

#[test]
fn the_basis_field_is_rejected() {
    let json = r#"{"mean_params": {"cell": {"basis": "linear"}}}"#;
    let err = serde_json::from_str::<Config>(json).unwrap_err();
    assert!(err.to_string().contains("basis"), "{err}");
}
