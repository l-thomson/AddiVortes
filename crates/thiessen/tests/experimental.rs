//! A build without the `experimental` feature rejects configurations that
//! name an experimental option.

#![cfg(not(feature = "experimental"))]

use thiessen::Config;

#[test]
fn unknown_config_field_is_rejected() {
    let err = serde_json::from_str::<Config>(r#"{"m": 5, "membership": "soft"}"#).unwrap_err();
    assert!(err.to_string().contains("membership"), "{err}");
}

#[test]
fn unknown_model_is_rejected() {
    let err = serde_json::from_str::<Config>(r#"{"model": "robust"}"#).unwrap_err();
    assert!(err.to_string().contains("robust"), "{err}");
}
