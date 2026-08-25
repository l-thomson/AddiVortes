//! The Laplace model at the fit boundary: the configuration surface, the
//! Laplace prediction semantics and persistence.

#![cfg(feature = "experimental")]

mod common;

use common::{fixture, SEED};
use thiessen::{fit, Config, Data, Fitted, Outcome};

/// The Gaussian fixture with three rows pushed far off the surface.
fn laplace_fixture() -> (Config, Data, Vec<f64>) {
    let (config, x, mut y) = fixture();
    for &row in &[5, 20, 35] {
        y[row] += 3.0;
    }
    (config.with_outcome(Outcome::laplace()), x, y)
}

/// The normal-mixture quantile the Gaussian model's interval uses, over
/// the draws of one row, for comparison against the Laplace mixture.
fn normal_mixture_quantile(fits: &[f64], sigmas: &[f64], p: f64) -> f64 {
    let cdf = |t: f64| {
        fits.iter()
            .zip(sigmas)
            .map(|(&f, &s)| 0.5 * libm::erfc(-((t - f) / s) * std::f64::consts::FRAC_1_SQRT_2))
            .sum::<f64>()
            / fits.len() as f64
    };
    let sigma_max = sigmas.iter().cloned().fold(0.0, f64::max);
    let (mut lo, mut hi) = (
        fits.iter().cloned().fold(f64::INFINITY, f64::min) - 40.0 * sigma_max,
        fits.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 40.0 * sigma_max,
    );
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[test]
fn outliers_change_the_chain_and_the_fit_round_trips() {
    let (config, x, y) = laplace_fixture();
    let gaussian = fit(
        &Config::new().with_m(15).with_burn_in(50).with_draws(60),
        &x,
        &y,
        SEED,
    )
    .unwrap();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    assert_eq!(fitted.model_name(), "laplace");
    assert_ne!(gaussian.sigma(), fitted.sigma());
    assert!(fitted.posterior().dfs().is_empty());
    let json = serde_json::to_string(&fitted).unwrap();
    let back: Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
    assert_eq!(
        fitted.prediction_interval(&x, 0.9).unwrap(),
        back.prediction_interval(&x, 0.9).unwrap()
    );
    assert_eq!(back.model_name(), "laplace");
}

/// The interval is the Laplace mixture, not the normal mixture with the
/// same scales: Laplace(0, b) reaches 0.975 at b ln 20, beyond 1.96 b,
/// so it is wider on both sides.
#[test]
fn the_prediction_interval_uses_laplace_tails() {
    let (config, x, y) = laplace_fixture();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    let intervals = fitted.prediction_interval(&x, 0.95).unwrap();
    let per_draw = fitted.predict_draws(&x).unwrap();
    let sigmas = fitted.sigma();
    for (row, interval) in intervals.iter().enumerate().take(6) {
        let fits: Vec<f64> = per_draw.iter().map(|draw| draw[row]).collect();
        let lower = normal_mixture_quantile(&fits, &sigmas, 0.025);
        let upper = normal_mixture_quantile(&fits, &sigmas, 0.975);
        assert!(
            interval.lower < lower,
            "row {row}: {} vs {lower}",
            interval.lower
        );
        assert!(
            interval.upper > upper,
            "row {row}: {} vs {upper}",
            interval.upper
        );
    }
}

#[test]
fn the_log_likelihood_is_the_laplace_density() {
    let (config, x, y) = laplace_fixture();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    let by_draw = fitted.log_likelihood(&x, &y).unwrap();
    let per_draw = fitted.predict_draws(&x).unwrap();
    let sigmas = fitted.sigma();
    for (d, draw) in by_draw.iter().enumerate() {
        for (i, &value) in draw.iter().enumerate() {
            let expected = -(2.0 * sigmas[d]).ln() - (y[i] - per_draw[d][i]).abs() / sigmas[d];
            assert!((value - expected).abs() < 1e-9, "{value} vs {expected}");
        }
    }
}

#[test]
fn predict_variance_is_twice_the_squared_scale() {
    let (config, x, y) = laplace_fixture();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    let variances = fitted.predict_variance(&x).unwrap();
    for (draw, sigma) in variances.iter().zip(fitted.sigma()) {
        let expected = 2.0 * sigma * sigma;
        assert!(draw.iter().all(|v| (v - expected).abs() < 1e-9 * expected));
    }
}

#[test]
fn a_variance_ensemble_is_rejected_for_identification() {
    let (config, x, y) = laplace_fixture();
    let err = fit(&config.with_m_var(5), &x, &y, SEED).unwrap_err();
    assert!(err.to_string().contains("identification"), "{err}");
    assert!(err.to_string().contains("laplace"), "{err}");
}

#[test]
fn the_sigma_prior_is_validated() {
    let (config, x, y) = laplace_fixture();
    let err = fit(&config.clone().with_nu(0.0), &x, &y, SEED).unwrap_err();
    assert!(err.to_string().contains("nu"), "{err}");
    let err = fit(&config.with_q(1.5), &x, &y, SEED).unwrap_err();
    assert!(err.to_string().contains("(0, 1)"), "{err}");
}

#[test]
fn the_outcome_serialises_in_snake_case_and_round_trips() {
    let (config, _, _) = laplace_fixture();
    let json = serde_json::to_string(&config).unwrap();
    assert!(
        json.contains(r#""outcome":{"laplace":{"nu":6.0,"q":0.85}}"#),
        "{json}"
    );
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(config, back);
    let bare: Config = serde_json::from_str(r#"{"outcome": {"laplace": {}}}"#).unwrap();
    assert_eq!(bare.outcome, Outcome::laplace());
}
