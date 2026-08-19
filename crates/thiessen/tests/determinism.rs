//! Same seed, same crate version, same target triple gives identical draws
//! (crate-root documentation, Reproducibility).

mod common;

use common::{fixture, SEED};

#[test]
fn same_seed_gives_identical_draws() {
    let (config, x, y) = fixture();
    let a = thiessen::fit(&config, &x, &y, SEED).unwrap();
    let b = thiessen::fit(&config, &x, &y, SEED).unwrap();
    assert_eq!(a.posterior().sigma_sq(), b.posterior().sigma_sq());
    assert_eq!(a.predict_draws(&x).unwrap(), b.predict_draws(&x).unwrap());
}

#[test]
fn different_seeds_give_different_draws() {
    let (config, x, y) = fixture();
    let a = thiessen::fit(&config, &x, &y, SEED).unwrap();
    let c = thiessen::fit(&config, &x, &y, SEED + 1).unwrap();
    assert_ne!(a.posterior().sigma_sq(), c.posterior().sigma_sq());
}
