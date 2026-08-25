//! Pooling the chains of a multi-chain run: `chain_seed` keys the chains
//! and `Fitted::pool` stacks their draws.

mod common;

use common::{fixture, probit_fixture, SEED};
use thiessen::{chain_seed, fit, fit_chains, Config, Data, Fitted, Sampler};

fn run_schedule(config: &Config, x: &Data, y: &[f64], k: usize) -> Sampler {
    let schedule = &config.general_params;
    let mut sampler = Sampler::new(config, x, y, chain_seed(SEED, k)).unwrap();
    for _ in 0..schedule.burn_in {
        sampler.step();
    }
    for _ in 0..schedule.draws {
        for _ in 0..schedule.thinning {
            sampler.step();
        }
        sampler.keep();
    }
    sampler
}

fn close(a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len());
    for (u, v) in a.iter().zip(b) {
        assert!((u - v).abs() <= 1e-12 * u.abs().max(1.0), "{u} against {v}");
    }
}

/// One chain pooled from its sampler equals the pooled pass over its
/// finished draws bit for bit.
fn one_chain_pools_exactly(config: &Config, x: &Data, y: &[f64]) {
    let finished = Fitted::pool(&[run_schedule(config, x, y, 0).finish().unwrap()], x, y).unwrap();

    let (pooled, fitted_values) =
        Fitted::pool_samplers(vec![run_schedule(config, x, y, 0)], x, y).unwrap();

    assert_eq!(pooled.sigma(), finished.sigma());
    assert_eq!(pooled.in_sample_rmse(), finished.in_sample_rmse());
    assert_eq!(fitted_values, finished.predict(x).unwrap());
}

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

    assert_eq!(pooled.n_draws(), 3 * config.general_params.draws);
    let stacked: Vec<f64> = parts.iter().flat_map(Fitted::sigma).collect();
    assert_eq!(pooled.sigma(), stacked);
    let draws = pooled.predict_draws(&x).unwrap();
    assert_eq!(draws.len(), 3 * config.general_params.draws);
    assert_eq!(draws[0], parts[0].predict_draws(&x).unwrap()[0]);
    assert_eq!(
        draws[config.general_params.draws],
        parts[1].predict_draws(&x).unwrap()[0]
    );
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
fn one_gaussian_chain_pools_from_its_sums_exactly() {
    let (config, x, y) = fixture();
    one_chain_pools_exactly(&config, &x, &y);
}

#[test]
fn one_probit_chain_pools_from_its_sums_exactly() {
    let (config, x, labels) = probit_fixture();
    one_chain_pools_exactly(&config, &x, &labels);
}

#[test]
fn one_heteroscedastic_chain_pools_from_its_sums_exactly() {
    let (config, x, y) = fixture();
    one_chain_pools_exactly(&config.with_m_var(5), &x, &y);
}

#[cfg(feature = "experimental")]
#[test]
fn one_linear_basis_chain_pools_from_its_sums_exactly() {
    let (config, x, y) = fixture();
    one_chain_pools_exactly(&config.with_basis(thiessen::Basis::Linear), &x, &y);
}

#[test]
fn a_thinned_schedule_pools_from_its_sums_exactly() {
    let (config, x, y) = fixture();
    one_chain_pools_exactly(&config.with_thinning(3).with_draws(7), &x, &y);
}

#[test]
fn pooling_samplers_equals_pooling_their_finished_chains() {
    let (config, x, y) = fixture();
    let finished = Fitted::pool(
        &[
            run_schedule(&config, &x, &y, 0).finish().unwrap(),
            run_schedule(&config, &x, &y, 1).finish().unwrap(),
        ],
        &x,
        &y,
    )
    .unwrap();

    let (pooled, fitted_values) = Fitted::pool_samplers(
        vec![
            run_schedule(&config, &x, &y, 0),
            run_schedule(&config, &x, &y, 1),
        ],
        &x,
        &y,
    )
    .unwrap();

    assert_eq!(pooled.sigma(), finished.sigma());
    assert_eq!(
        pooled.predict_draws(&x).unwrap(),
        finished.predict_draws(&x).unwrap()
    );
    close(&[pooled.in_sample_rmse()], &[finished.in_sample_rmse()]);
    close(&fitted_values, &pooled.predict(&x).unwrap());
}

#[test]
fn fit_chains_equals_the_pooled_single_fits() {
    let (config, x, y) = fixture();
    let parts = chains(&config, &x, &y, 2);
    let via_pool = Fitted::pool(&parts, &x, &y).unwrap();

    let (pooled, fitted_values) = fit_chains(&config, &x, &y, SEED, 2).unwrap();

    assert_eq!(pooled.sigma(), via_pool.sigma());
    assert_eq!(
        pooled.predict_draws(&x).unwrap(),
        via_pool.predict_draws(&x).unwrap()
    );
    close(&[pooled.in_sample_rmse()], &[via_pool.in_sample_rmse()]);
    close(&fitted_values, &via_pool.predict(&x).unwrap());
    assert!(fit_chains(&config, &x, &y, SEED, 0).is_err());
}

#[test]
fn finish_reports_the_same_rmse_as_before_the_sums() {
    let (config, x, y) = fixture();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    let by_pass = Fitted::pool(std::slice::from_ref(&fitted), &x, &y).unwrap();
    assert!((fitted.in_sample_rmse() - by_pass.in_sample_rmse()).abs() < 1e-12);
}

#[test]
fn a_sampler_with_no_kept_draw_does_not_pool() {
    let (config, x, y) = fixture();
    let sampler = Sampler::new(&config, &x, &y, SEED).unwrap();
    assert!(Fitted::pool_samplers(vec![sampler], &x, &y).is_err());
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
