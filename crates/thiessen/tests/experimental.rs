//! A build without the `experimental` feature carries a gated value
//! through the deserialiser, which states the shape, and reports it from
//! `Config::validate`, which states the policy, as `RequiresFeature`
//! naming the value and the feature.
//!
//! `gated_surface.rs` runs the same configurations under the feature and
//! `published_defaults.rs` the published default of every gated field.

#![cfg(not(feature = "experimental"))]

mod common;

use common::GATED_CONFIGS;
use thiessen::{Config, Error, Fitted};

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

/// `Fitted::load` reports a saved fit's gated configuration as the typed
/// error, where the `Deserialize` impl has only the format's text; a
/// bandwidth kept without the membership that draws one is a broken
/// payload in every build.
#[test]
fn a_saved_fit_naming_a_gated_item_names_the_feature() {
    let (config, x, y) = common::fixture();
    let config = config.with_burn_in(2).with_draws(2);
    let fitted = thiessen::fit(&config, &x, &y, common::SEED).unwrap();
    let mut saved = serde_json::to_value(&fitted).unwrap();
    saved["config"]["mean_params"]["geometry"]["membership"] = serde_json::json!({"soft": {}});
    assert!(matches!(
        Fitted::load(&saved),
        Err(Error::RequiresFeature { ref item, .. }) if item.contains("soft")
    ));
    let text = serde_json::to_string(&saved).unwrap();
    let err = serde_json::from_str::<Fitted>(&text).unwrap_err();
    assert!(err.to_string().contains("experimental"), "{err}");

    saved["config"]["mean_params"]["geometry"] = serde_json::json!({});
    saved["posterior"]["tessellations"][0][0]["tau"] = serde_json::json!(0.2);
    assert!(matches!(
        Fitted::load(&saved),
        Err(Error::InvalidSavedModel { ref reason }) if reason.contains("soft membership")
    ));
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
        thiessen::fit_aft_chains_with_threads(&config, &x, &ones, &events, common::SEED, 1, 1)
            .map(drop),
        "aft",
    );
    gated(
        thiessen::fit_interval_censored_chains_with_threads(
            &config,
            &x,
            &ones,
            &ones,
            common::SEED,
            1,
            1,
        )
        .map(drop),
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
