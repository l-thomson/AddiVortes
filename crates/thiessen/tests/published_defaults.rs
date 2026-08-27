//! The published default of every gated field is accepted in every build
//! and stays out of the serialised form: refusing it would refuse a user
//! the default. `experimental.rs` and `gated_surface.rs` read the same
//! list for the values the gate covers.

mod common;

use common::PUBLISHED_DEFAULTS;
use thiessen::Config;

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
