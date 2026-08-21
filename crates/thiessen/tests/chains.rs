//! Pooling the chains of a multi-chain run: `chain_seed` keys the chains
//! and `Fitted::pool` stacks their draws.

mod common;

use common::{fixture, SEED};
use thiessen::{chain_seed, fit, Config, Data, Fitted};

fn chains(config: &Config, x: &Data, y: &[f64], n: usize) -> Vec<Fitted> {
    (0..n)
        .map(|k| fit(config, x, y, chain_seed(SEED, k)).unwrap())
        .collect()
}

#[test]
fn one_chain_pools_to_itself() {
    let (config, x, y) = fixture();
    let single = chains(&config, &x, &y, 1);

    let pooled = Fitted::pool(&single, &x, &y).unwrap();

    assert_eq!(pooled.n_draws(), single[0].n_draws());
    assert_eq!(pooled.sigma(), single[0].sigma());
    assert_eq!(pooled.predict(&x).unwrap(), single[0].predict(&x).unwrap());
    assert!((pooled.in_sample_rmse() - single[0].in_sample_rmse()).abs() < 1e-12);
}

#[test]
fn pooling_stacks_the_draws_in_chain_order() {
    let (config, x, y) = fixture();
    let parts = chains(&config, &x, &y, 3);

    let pooled = Fitted::pool(&parts, &x, &y).unwrap();

    assert_eq!(pooled.n_draws(), 3 * config.draws);
    let stacked: Vec<f64> = parts.iter().flat_map(Fitted::sigma).collect();
    assert_eq!(pooled.sigma(), stacked);
    let draws = pooled.predict_draws(&x).unwrap();
    assert_eq!(draws.len(), 3 * config.draws);
    assert_eq!(draws[0], parts[0].predict_draws(&x).unwrap()[0]);
    assert_eq!(draws[config.draws], parts[1].predict_draws(&x).unwrap()[0]);
}

#[test]
fn chains_differ_and_the_pooled_mean_is_their_average() {
    let (config, x, y) = fixture();
    let parts = chains(&config, &x, &y, 2);
    assert_ne!(parts[0].sigma(), parts[1].sigma());

    let pooled = Fitted::pool(&parts, &x, &y).unwrap();

    let a = parts[0].predict(&x).unwrap();
    let b = parts[1].predict(&x).unwrap();
    for (i, value) in pooled.predict(&x).unwrap().iter().enumerate() {
        assert!((value - 0.5 * (a[i] + b[i])).abs() < 1e-9);
    }
}

#[test]
fn a_pooled_fit_round_trips_through_serde() {
    let (config, x, y) = fixture();
    let pooled = Fitted::pool(&chains(&config, &x, &y, 2), &x, &y).unwrap();

    let json = serde_json::to_string(&pooled).unwrap();
    let loaded: Fitted = serde_json::from_str(&json).unwrap();

    assert_eq!(loaded.n_draws(), pooled.n_draws());
    assert_eq!(loaded.predict(&x).unwrap(), pooled.predict(&x).unwrap());
}

#[test]
fn chains_of_different_configurations_do_not_pool() {
    let (config, x, y) = fixture();
    let first = fit(&config, &x, &y, SEED).unwrap();
    let second = fit(&config.clone().with_m(10), &x, &y, SEED).unwrap();

    let error = Fitted::pool(&[first, second], &x, &y).unwrap_err();

    assert!(matches!(error, thiessen::Error::MismatchedChains { .. }));
    assert!(Fitted::pool(&[], &x, &y).is_err());
}
