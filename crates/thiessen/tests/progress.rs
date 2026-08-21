//! `fit_with_progress` reports every sweep and leaves the draws unchanged.

mod common;

use common::{fixture, SEED};

#[test]
fn every_sweep_is_reported_once_in_order() {
    let (config, x, y) = fixture();
    let config = config.with_thinning(2);
    let total = config.burn_in + config.draws * config.thinning;
    let mut seen = Vec::new();
    thiessen::fit_with_progress(&config, &x, &y, SEED, |completed, reported| {
        seen.push((completed, reported));
    })
    .unwrap();

    assert_eq!(seen.len(), total);
    assert_eq!(seen.first(), Some(&(1, total)));
    assert_eq!(seen.last(), Some(&(total, total)));
    assert!(seen.windows(2).all(|w| w[1].0 == w[0].0 + 1));
}

#[test]
fn reporting_does_not_change_the_draws() {
    let (config, x, y) = fixture();
    let plain = thiessen::fit(&config, &x, &y, SEED).unwrap();
    let reported = thiessen::fit_with_progress(&config, &x, &y, SEED, |_, _| {}).unwrap();

    assert_eq!(
        plain.posterior().sigma_sq(),
        reported.posterior().sigma_sq()
    );
    assert_eq!(
        plain.predict_draws(&x).unwrap(),
        reported.predict_draws(&x).unwrap()
    );
}
