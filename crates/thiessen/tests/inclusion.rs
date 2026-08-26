//! The weighted inclusion prior at the fit boundary: the equal-weight
//! equivalence with the uniform prior, zero-weight exclusion, and the
//! configuration surface.

#![cfg(feature = "experimental")]

mod common;

use common::{fixture, SEED};
use thiessen::{fit, Config, Inclusion};

fn weighted(weights: Vec<f64>) -> Inclusion {
    Inclusion::Weighted { weights }
}

fn dart() -> Inclusion {
    Inclusion::Dart {
        a: 0.5,
        b: 1.0,
        rho: None,
    }
}

#[test]
fn equal_weights_reproduce_the_uniform_chain() {
    let (config, x, y) = fixture();
    let uniform = fit(&config.clone(), &x, &y, SEED).unwrap();
    let fitted = fit(
        &config.with_inclusion(weighted(vec![0.5, 0.5])),
        &x,
        &y,
        SEED,
    )
    .unwrap();
    assert_eq!(uniform.sigma(), fitted.sigma());
    assert_eq!(
        uniform.predict_draws(&x).unwrap(),
        fitted.predict_draws(&x).unwrap()
    );
}

#[test]
fn a_zero_weight_excludes_the_column() {
    let (config, x, y) = fixture();
    let fitted = fit(
        &config.with_inclusion(weighted(vec![1.0, 0.0])),
        &x,
        &y,
        SEED,
    )
    .unwrap();
    assert_eq!(fitted.variable_inclusion_proportions()[1], 0.0);
    for draw in fitted.posterior().tessellations() {
        for t in draw {
            assert_eq!(t.dims(), [0]);
        }
    }
}

#[test]
fn unequal_weights_change_the_chain_and_round_trip() {
    let (config, x, y) = fixture();
    let uniform = fit(&config.clone(), &x, &y, SEED).unwrap();
    let fitted = fit(
        &config.with_inclusion(weighted(vec![0.75, 0.25])),
        &x,
        &y,
        SEED,
    )
    .unwrap();
    assert_ne!(uniform.sigma(), fitted.sigma());
    let json = serde_json::to_string(&fitted).unwrap();
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
}

#[test]
fn a_wrong_length_is_a_fit_error() {
    let (config, x, y) = fixture();
    assert!(fit(
        &config.with_inclusion(weighted(vec![1.0, 1.0, 1.0])),
        &x,
        &y,
        SEED
    )
    .is_err());
}

#[test]
fn the_inclusion_serialises_compactly_and_round_trips() {
    let json = serde_json::to_string(&Config::new()).unwrap();
    assert!(!json.contains("inclusion"), "{json}");
    let config = Config::new().with_inclusion(weighted(vec![0.75, 0.25]));
    let json = serde_json::to_string(&config).unwrap();
    assert!(
        json.contains(r#""inclusion":{"weighted":{"weights":[0.75,0.25]}}"#),
        "{json}"
    );
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(back, config);
}

#[test]
fn dart_defaults_parse_and_round_trip() {
    let json = r#"{"mean_params": {"structure": {"inclusion": {"dart": {}}}}}"#;
    let config: Config = serde_json::from_str(json).unwrap();
    let expected = Inclusion::Dart {
        a: 0.5,
        b: 1.0,
        rho: None,
    };
    assert_eq!(config.mean_params.structure.inclusion, expected);
    let config = Config::new().with_inclusion(Inclusion::Dart {
        a: 2.0,
        b: 2.0,
        rho: Some(3.0),
    });
    let json = serde_json::to_string(&config).unwrap();
    assert!(
        json.contains(r#""inclusion":{"dart":{"a":2.0,"b":2.0,"rho":3.0}}"#),
        "{json}"
    );
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(back, config);
}

#[test]
fn a_dart_fit_runs_and_round_trips() {
    let (config, x, y) = fixture();
    let fitted = fit(&config.with_inclusion(dart()), &x, &y, SEED).unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
}

/// The response of the fixture depends on both columns; rebuilt on the
/// first column alone, DART concentrates inclusion there.
#[test]
fn dart_concentrates_on_the_informative_column() {
    let (config, x, _) = fixture();
    let y: Vec<f64> = (0..x.n_rows())
        .map(|i| {
            let v = x.row(i)[0];
            3.0 * (v - 0.4) * (v - 0.4)
        })
        .collect();
    let noisy: Vec<f64> = y
        .iter()
        .enumerate()
        .map(|(i, v)| v + 0.05 * (((i * 29) % 17) as f64 / 16.0 - 0.5))
        .collect();
    // The default omega resolves to p here and every tessellation then
    // uses both columns; concentration needs a varying subset, so the
    // test lowers omega.
    let fitted = fit(
        &config.with_omega(0.5).with_inclusion(dart()),
        &x,
        &noisy,
        SEED,
    )
    .unwrap();
    let proportions = fitted.variable_inclusion_proportions();
    assert!(proportions[0] > 2.0 * proportions[1], "{proportions:?}");
}

#[test]
fn a_dart_fit_keeps_its_weights_and_concentration() {
    let (config, x, y) = fixture();
    let config = config.with_inclusion(dart());
    let fitted = fit(&config, &x, &y, SEED).unwrap();

    let weights = fitted.inclusion_weight_draws();
    assert_eq!(weights.len(), fitted.n_draws());
    assert_eq!(fitted.concentration_draws().len(), fitted.n_draws());
    for draw in weights {
        assert_eq!(draw.len(), x.n_cols());
        assert!((draw.iter().sum::<f64>() - 1.0).abs() < 1e-9, "{draw:?}");
    }
    assert!(fitted.concentration_draws().iter().all(|t| *t > 0.0));
}

/// The weights kept are those the sampler used, not merely of the right
/// shape: the same schedule driven by hand records the same values.
#[test]
fn the_kept_weights_are_the_ones_the_sampler_drew() {
    let (config, x, y) = fixture();
    let config = config.with_inclusion(dart());
    let fitted = fit(&config, &x, &y, SEED).unwrap();

    let mut sampler = thiessen::Sampler::new(&config, &x, &y, SEED).unwrap();
    let schedule = &config.general_params;
    for _ in 0..schedule.burn_in {
        sampler.step();
    }
    let mut expected = Vec::new();
    for _ in 0..schedule.draws {
        for _ in 0..schedule.thinning {
            sampler.step();
        }
        let (s, theta) = sampler.inclusion_state().expect("dart state");
        expected.push((s.to_vec(), theta));
        sampler.keep();
    }

    assert_eq!(
        fitted.inclusion_weight_draws(),
        expected.iter().map(|(s, _)| s.clone()).collect::<Vec<_>>()
    );
    assert_eq!(
        fitted.concentration_draws(),
        expected.iter().map(|(_, t)| *t).collect::<Vec<_>>()
    );
}

#[test]
fn another_inclusion_prior_keeps_neither() {
    let (config, x, y) = fixture();
    let uniform = fit(&config.clone(), &x, &y, SEED).unwrap();
    assert!(uniform.inclusion_weight_draws().is_empty());
    assert!(uniform.concentration_draws().is_empty());

    let weighted = fit(
        &config.with_inclusion(weighted(vec![0.5, 0.5])),
        &x,
        &y,
        SEED,
    )
    .unwrap();
    assert!(weighted.inclusion_weight_draws().is_empty());
    assert!(weighted.concentration_draws().is_empty());
}

#[test]
fn the_dart_state_survives_a_round_trip_and_a_payload_without_it() {
    let (config, x, y) = fixture();
    let fitted = fit(&config.clone().with_inclusion(dart()), &x, &y, SEED).unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    assert!(json.contains(r#""inclusion_weights":"#), "{json}");
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(
        back.inclusion_weight_draws(),
        fitted.inclusion_weight_draws()
    );
    assert_eq!(back.concentration_draws(), fitted.concentration_draws());

    // A payload written before the fields existed carries neither name.
    let uniform = fit(&config, &x, &y, SEED).unwrap();
    let json = serde_json::to_string(&uniform).unwrap();
    assert!(!json.contains("inclusion_weights"), "{json}");
    assert!(!json.contains("concentration"), "{json}");
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(back.predict(&x).unwrap(), uniform.predict(&x).unwrap());
}
