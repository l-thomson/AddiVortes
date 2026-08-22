//! The linear cell basis at the fit boundary: the configuration surface,
//! the fit-time scaling requirement, and persistence of the slopes.

#![cfg(feature = "experimental")]

mod common;

use common::{fixture, SEED};
use thiessen::{fit, Basis, Config, Metric};

fn linear(config: Config) -> Config {
    config.with_basis(Basis::Linear)
}

#[test]
fn the_linear_basis_changes_the_chain_and_round_trips() {
    let (config, x, y) = fixture();
    let constant = fit(&config.clone(), &x, &y, SEED).unwrap();
    let fitted = fit(&linear(config), &x, &y, SEED).unwrap();
    assert_ne!(constant.sigma(), fitted.sigma());
    let json = serde_json::to_string(&fitted).unwrap();
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
}

#[test]
fn the_slopes_are_kept_and_predictions_tilt() {
    let (config, x, y) = fixture();
    let fitted = fit(&linear(config), &x, &y, SEED).unwrap();
    let draw = &fitted.posterior().tessellations()[0][0];
    let json = serde_json::to_string(draw).unwrap();
    assert!(json.contains(r#""betas":["#), "{json}");
}

#[test]
fn a_linear_fit_recovers_a_tilted_function_better() {
    let n = 60;
    let rows: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            [
                i as f64 / (n - 1) as f64,
                ((i * 23) % n) as f64 / (n - 1) as f64,
            ]
        })
        .collect();
    let y: Vec<f64> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| 2.0 * r[0] - r[1] + 0.02 * (((i * 29) % 17) as f64 / 16.0 - 0.5))
        .collect();
    let x = thiessen::Data::from_rows(&rows).unwrap();
    let config = Config::new().with_m(5).with_burn_in(100).with_draws(100);
    let constant = fit(&config.clone(), &x, &y, SEED).unwrap();
    let tilted = fit(&linear(config), &x, &y, SEED).unwrap();
    assert!(
        tilted.in_sample_rmse() < constant.in_sample_rmse(),
        "{} vs {}",
        tilted.in_sample_rmse(),
        constant.in_sample_rmse()
    );
}

#[test]
fn the_linear_basis_needs_scaled_columns() {
    let (config, x, y) = fixture();
    let rows: Vec<Vec<f64>> = (0..x.n_rows())
        .map(|i| vec![x.row(i)[0], (x.row(i)[1] * 3.0).round()])
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let coded = thiessen::Data::from_rows(&refs).unwrap();
    let err = fit(
        &linear(config).with_metric(vec![Metric::Euclidean, Metric::Categorical]),
        &coded,
        &y,
        SEED,
    )
    .unwrap_err();
    assert!(err.to_string().contains("cell.basis"), "{err}");
}

#[test]
fn the_variance_slot_keeps_the_constant_basis() {
    let json = r#"{"variance_params": {"tessellations": 2, "cell": {"basis": "linear"}}}"#;
    let config: Config = serde_json::from_str(json).unwrap();
    assert!(config.validate().is_err());
}

#[test]
fn the_basis_serialises_compactly_and_round_trips() {
    let json = serde_json::to_string(&Config::new()).unwrap();
    assert!(!json.contains("basis"), "{json}");
    let config = Config::new().with_basis(Basis::Linear);
    let json = serde_json::to_string(&config).unwrap();
    assert!(json.contains(r#""cell":{"basis":"linear"}"#), "{json}");
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(back, config);
}
