//! The draws and the predictions do not depend on the thread count.

mod common;

use common::{fixture, SEED};
use thiessen::{fit, fit_chains, fit_chains_with_threads, Sampler};

#[test]
fn threaded_chains_equal_the_chains_run_in_turn() {
    let (config, x, y) = fixture();
    let (serial, serial_values) = fit_chains(&config, &x, &y, SEED, 3).unwrap();
    for threads in [2, 3, 8] {
        let (threaded, values) =
            fit_chains_with_threads(&config, &x, &y, SEED, 3, threads).unwrap();
        assert_eq!(threaded, serial, "threads = {threads}");
        assert_eq!(values, serial_values);
        assert_eq!(threaded.threads(), threads);
    }
    assert_eq!(serial.threads(), 1);
}

#[test]
fn advance_all_matches_the_samplers_advanced_alone() {
    let (config, x, y) = fixture();
    let mut alone: Vec<Sampler> = (0..3)
        .map(|k| Sampler::new(&config, &x, &y, thiessen::chain_seed(SEED, k)).unwrap())
        .collect();
    for sampler in &mut alone {
        for _ in 0..4 {
            sampler.step();
        }
        for _ in 0..3 {
            sampler.step();
            sampler.step();
            sampler.keep();
        }
    }
    let mut together: Vec<Sampler> = (0..3)
        .map(|k| Sampler::new(&config, &x, &y, thiessen::chain_seed(SEED, k)).unwrap())
        .collect();
    let mut refs: Vec<&mut Sampler> = together.iter_mut().collect();
    Sampler::advance_all(&mut refs, 4, 3, 2, 2);
    for (a, t) in alone.into_iter().zip(together) {
        assert_eq!(a.finish().unwrap(), t.finish().unwrap());
    }
}

#[test]
fn predictions_do_not_depend_on_the_thread_count() {
    let (config, x, y) = fixture();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    let one = fitted.predict_draws(&x).unwrap();
    let mean = fitted.predict(&x).unwrap();
    let interval = fitted.prediction_interval(&x, 0.9).unwrap();
    let credible = fitted.credible_interval(&x, 0.9).unwrap();
    let quantiles = fitted.predict_quantiles(&x, &[0.1, 0.5, 0.9]).unwrap();
    let mut threaded = fitted.clone();
    // Counts that divide the rows unevenly and one above the row count.
    for threads in [2, 5, 7, 1000] {
        threaded.set_threads(threads);
        assert_eq!(threaded.threads(), threads);
        assert_eq!(
            threaded.predict_draws(&x).unwrap(),
            one,
            "threads = {threads}"
        );
        assert_eq!(threaded.predict(&x).unwrap(), mean);
        assert_eq!(threaded.prediction_interval(&x, 0.9).unwrap(), interval);
        assert_eq!(threaded.credible_interval(&x, 0.9).unwrap(), credible);
        assert_eq!(
            threaded.predict_quantiles(&x, &[0.1, 0.5, 0.9]).unwrap(),
            quantiles
        );
    }
    threaded.set_threads(0);
    assert_eq!(threaded.threads(), 1);
}

#[test]
fn the_heteroscedastic_variance_does_not_depend_on_the_thread_count() {
    let (config, x, y) = fixture();
    let fitted = fit(&config.with_m_var(5), &x, &y, SEED).unwrap();
    let one = fitted.predict_variance(&x).unwrap();
    let mut threaded = fitted.clone();
    threaded.set_threads(3);
    assert_eq!(threaded.predict_variance(&x).unwrap(), one);
}

#[test]
fn the_thread_count_is_not_persisted() {
    let (config, x, y) = fixture();
    let (fitted, _) = fit_chains_with_threads(&config, &x, &y, SEED, 2, 2).unwrap();
    let json = serde_json::to_string(&fitted).unwrap();
    let loaded: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.threads(), 1);
    assert_eq!(loaded, fitted);
}
