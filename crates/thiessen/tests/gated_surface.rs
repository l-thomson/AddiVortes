//! The configuration surface a build without the `experimental` feature
//! gates, run under the feature: every configuration `experimental.rs`
//! rejects there is accepted here. The two tests read one list, so the
//! builds cannot disagree about what the gate covers.

#![cfg(feature = "experimental")]

mod common;

use common::GATED_CONFIGS;
use thiessen::Config;

#[test]
fn every_gated_item_is_accepted_under_the_feature() {
    for (json, name) in GATED_CONFIGS {
        let config: Config = serde_json::from_str(json)
            .unwrap_or_else(|error| panic!("{name}: {json} should deserialise: {error}"));
        config
            .validate()
            .unwrap_or_else(|error| panic!("{name}: {json} should validate: {error}"));
    }
}
