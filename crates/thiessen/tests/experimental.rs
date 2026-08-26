//! A build without the `experimental` feature carries a gated value
//! through the deserialiser, which states the shape, and reports it from
//! `Config::validate`, which states the policy, as `RequiresFeature`
//! naming the value and the feature. The published default of every
//! gated field is accepted: refusing it would refuse a user the default.
//!
//! `gated_surface.rs` runs the same configurations under the feature.

#![cfg(not(feature = "experimental"))]

mod common;

use common::{GATED_CONFIGS, PUBLISHED_DEFAULTS};
use thiessen::{Config, Error};

#[test]
fn every_gated_item_names_itself_and_the_feature() {
    for (json, name) in GATED_CONFIGS {
        let config: Config = serde_json::from_str(json)
            .unwrap_or_else(|error| panic!("{json} should deserialise: {error}"));
        match config.validate() {
            Err(Error::RequiresFeature { item, feature }) => {
                assert_eq!(feature, "experimental");
                assert!(item.contains(name), "{item} should name {name}");
            }
            other => panic!("{json} should need the feature, got {other:?}"),
        }
    }
}

#[test]
fn the_published_defaults_are_accepted_and_stay_out_of_the_form() {
    for (json, field) in PUBLISHED_DEFAULTS {
        let config: Config = serde_json::from_str(json)
            .unwrap_or_else(|error| panic!("{json} should deserialise: {error}"));
        config
            .validate()
            .unwrap_or_else(|error| panic!("{json} should validate: {error}"));
        let back = serde_json::to_string(&config).unwrap();
        assert!(!back.contains(field), "{back}");
    }
}

/// A gated value on the variance slot is reported as one on the mean
/// slot is: the slot is not a way past the gate.
#[test]
fn a_gated_value_on_the_variance_slot_is_reported() {
    let json = r#"{"variance_params": {"tessellations": 4,
        "structure": {"inclusion": {"dart": {}}}}}"#;
    let config: Config = serde_json::from_str(json).unwrap();
    assert!(matches!(
        config.validate(),
        Err(Error::RequiresFeature { .. })
    ));
}

#[test]
fn an_unknown_outcome_is_still_a_deserialisation_error() {
    let err = serde_json::from_str::<Config>(r#"{"outcome": {"robust": {}}}"#).unwrap_err();
    assert!(err.to_string().contains("robust"), "{err}");
}

#[test]
fn an_unknown_field_is_still_a_deserialisation_error() {
    let json = r#"{"mean_params": {"geometry": {"bandwidth": 0.2}}}"#;
    let err = serde_json::from_str::<Config>(json).unwrap_err();
    assert!(err.to_string().contains("bandwidth"), "{err}");
}

#[test]
fn a_saved_bandwidth_names_the_feature() {
    let json = r#"{"centres":[0.1],"dims":[0],"mus":[1.0],"tau":0.2}"#;
    let err = serde_json::from_str::<thiessen::Tessellation>(json).unwrap_err();
    assert!(err.to_string().contains("experimental"), "{err}");
}

/// The entry points of the gated models exist in every build and report
/// the model's own name, so a binding wraps one surface.
#[test]
fn every_gated_entry_point_names_the_outcome_and_the_feature() {
    let (config, x, y) = common::fixture();
    let config = config.with_burn_in(2).with_draws(2);
    let fitted = thiessen::fit(&config, &x, &y, common::SEED).unwrap();
    let mut sampler = thiessen::Sampler::new(&config, &x, &y, common::SEED).unwrap();
    let ones = vec![1.0; y.len()];
    let events = vec![true; y.len()];
    let gated = |result: Result<(), Error>, name: &str| match result {
        Err(Error::RequiresFeature { item, feature }) => {
            assert_eq!(feature, "experimental");
            assert!(item.contains(name), "{item} should name {name}");
        }
        other => panic!("{name} should need the feature, got {other:?}"),
    };
    gated(
        thiessen::fit_aft(&config, &x, &ones, &events, common::SEED).map(drop),
        "aft",
    );
    gated(
        thiessen::fit_interval_censored(&config, &x, &ones, &ones, common::SEED).map(drop),
        "interval_censored",
    );
    gated(
        thiessen::Sampler::aft(&config, &x, &ones, &events, common::SEED).map(drop),
        "aft",
    );
    gated(
        thiessen::Sampler::interval_censored(&config, &x, &ones, &ones, common::SEED).map(drop),
        "interval_censored",
    );
    gated(sampler.set_aft_response(&ones, &events), "aft");
    gated(
        sampler.set_interval_censored_response(&ones, &ones),
        "interval_censored",
    );
    gated(
        fitted.log_likelihood_survival(&x, &ones, &events).map(drop),
        "aft",
    );
    gated(
        fitted
            .log_likelihood_interval_censored(&x, &ones, &ones)
            .map(drop),
        "interval_censored",
    );
    gated(
        fitted.predict_category_probabilities(&x).map(drop),
        "ordinal",
    );
    assert!(fitted.cutpoint_draws().is_empty());
    assert!(sampler.cutpoints().is_empty());
    assert_eq!(sampler.student_df(), None);
    assert_eq!(sampler.inclusion_state(), None);
}
