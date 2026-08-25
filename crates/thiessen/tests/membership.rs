//! Soft membership at the fit boundary: the hard default's equivalence
//! with the plain chain, the configuration surface, and persistence of
//! the bandwidth.

#![cfg(feature = "experimental")]

mod common;

use common::{fixture, probit_fixture, SEED};
use thiessen::{fit, Basis, Config, Membership};

fn soft(config: Config) -> Config {
    config.with_membership(Membership::soft())
}

#[test]
fn hard_membership_reproduces_the_plain_chain() {
    let (config, x, y) = fixture();
    let plain = fit(&config.clone(), &x, &y, SEED).unwrap();
    let fitted = fit(&config.with_membership(Membership::Hard), &x, &y, SEED).unwrap();
    assert_eq!(plain.sigma(), fitted.sigma());
    assert_eq!(
        plain.predict_draws(&x).unwrap(),
        fitted.predict_draws(&x).unwrap()
    );
}

#[test]
fn soft_membership_changes_the_chain_and_round_trips() {
    let (config, x, y) = fixture();
    let hard = fit(&config.clone(), &x, &y, SEED).unwrap();
    let fitted = fit(&soft(config), &x, &y, SEED).unwrap();
    assert_ne!(hard.sigma(), fitted.sigma());
    let json = serde_json::to_string(&fitted).unwrap();
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
}

#[test]
fn the_bandwidth_is_kept() {
    let (config, x, y) = fixture();
    let fitted = fit(&soft(config), &x, &y, SEED).unwrap();
    let draw = &fitted.posterior().tessellations()[0][0];
    assert!(draw.bandwidth().is_some_and(|tau| tau > 0.0));
    let json = serde_json::to_string(draw).unwrap();
    assert!(json.contains(r#""tau":"#), "{json}");
}

#[test]
fn the_bandwidths_are_reachable_a_row_per_draw() {
    let (config, x, y) = fixture();
    let m = config
        .mean_params
        .tessellations
        .expect("the fixture sets m");
    let fitted = fit(&soft(config), &x, &y, SEED).unwrap();

    let bandwidths = fitted.bandwidth_draws();
    assert_eq!(bandwidths.len(), fitted.n_draws());
    for (d, draw) in bandwidths.iter().enumerate() {
        assert_eq!(draw.len(), m);
        assert!(draw.iter().all(|tau| *tau > 0.0), "{draw:?}");
        let kept: Vec<f64> = fitted.posterior().tessellations()[d]
            .iter()
            .map(|t| t.bandwidth().expect("a soft tessellation carries tau"))
            .collect();
        assert_eq!(*draw, kept);
    }
}

#[test]
fn hard_membership_keeps_no_bandwidth() {
    let (config, x, y) = fixture();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    assert!(fitted.bandwidth_draws().is_empty());
}

#[test]
fn soft_membership_composes_with_the_probit_model() {
    let (config, x, labels) = probit_fixture();
    let fitted = fit(&soft(config), &x, &labels, SEED).unwrap();
    assert!(fitted
        .predict(&x)
        .unwrap()
        .iter()
        .all(|p| (0.0..=1.0).contains(p)));
}

#[test]
fn soft_membership_needs_a_constant_spread() {
    let (config, x, y) = fixture();
    let err = fit(&soft(config.with_m_var(5)), &x, &y, SEED).unwrap_err();
    assert!(err.to_string().contains("membership"), "{err}");
}

#[test]
fn soft_membership_takes_the_constant_basis() {
    let (config, x, y) = fixture();
    let err = fit(&soft(config.with_basis(Basis::Linear)), &x, &y, SEED).unwrap_err();
    assert!(err.to_string().contains("cell.basis"), "{err}");
}

#[test]
fn the_bandwidth_rate_must_be_positive() {
    let (config, x, y) = fixture();
    let err = fit(
        &config.with_membership(Membership::Soft { rate: 0.0 }),
        &x,
        &y,
        SEED,
    )
    .unwrap_err();
    assert!(err.to_string().contains("membership.rate"), "{err}");
}

#[test]
fn the_membership_serialises_compactly_and_round_trips() {
    let (config, _, _) = fixture();
    let hard = serde_json::to_string(&config).unwrap();
    assert!(!hard.contains("membership"), "{hard}");
    let config = soft(config);
    let json = serde_json::to_string(&config).unwrap();
    assert!(
        json.contains(r#""membership":{"soft":{"rate":10.0}}"#),
        "{json}"
    );
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(config, back);
}
