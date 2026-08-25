//! The Student-t model at the fit boundary: agreement with the Gaussian
//! model at large df, the configuration surface, the t prediction
//! semantics and persistence.

#![cfg(feature = "experimental")]

mod common;

use common::{fixture, SEED};
use thiessen::{fit, Config, Data, Error, Fitted, Outcome};

/// The Gaussian fixture with three rows pushed far off the surface,
/// the outliers the weights are there to discount.
fn student_t_fixture(df: f64) -> (Config, Data, Vec<f64>) {
    let (config, x, mut y) = fixture();
    for &row in &[5, 20, 35] {
        y[row] += 3.0;
    }
    (config.with_outcome(Outcome::student_t(df)), x, y)
}

/// The normal-mixture quantile the Gaussian model's interval uses, over
/// the draws of one row, for comparison against the t mixture.
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

/// At large df the weights concentrate at 1 and the posterior is the
/// Gaussian model's: the posterior-mean predictions and the mean sigma
/// agree within Monte Carlo error, a chain apart.
#[test]
fn large_df_reproduces_the_gaussian_posterior_within_monte_carlo_error() {
    let (config, x, y) = fixture();
    let config = config.with_burn_in(200).with_draws(400);
    let gaussian = fit(&config, &x, &y, SEED).unwrap();
    let student = fit(&config.with_outcome(Outcome::student_t(1e6)), &x, &y, SEED).unwrap();
    assert_eq!(student.model_name(), "student_t");
    let range = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - y.iter().cloned().fold(f64::INFINITY, f64::min);
    let (a, b) = (gaussian.predict(&x).unwrap(), student.predict(&x).unwrap());
    let mean_gap = a.iter().zip(&b).map(|(u, v)| (u - v).abs()).sum::<f64>() / a.len() as f64;
    assert!(mean_gap < 0.03 * range, "{mean_gap} against range {range}");
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (sa, sb) = (mean(&gaussian.sigma()), mean(&student.sigma()));
    assert!((sa - sb).abs() < 0.15 * sa, "{sa} vs {sb}");
}

#[test]
fn outliers_change_the_chain_and_the_fit_round_trips() {
    let (config, x, y) = student_t_fixture(4.0);
    let gaussian = fit(
        &Config::new().with_m(15).with_burn_in(50).with_draws(60),
        &x,
        &y,
        SEED,
    )
    .unwrap();
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    assert_ne!(gaussian.sigma(), fitted.sigma());
    assert!(fitted.posterior().dfs().is_empty());
    let json = serde_json::to_string(&fitted).unwrap();
    let back: Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
    assert_eq!(back.model_name(), "student_t");
}

/// The interval is the t mixture, not the normal mixture with the same
/// scales: at df = 3 it is wider on both sides.
#[test]
fn the_prediction_interval_uses_t_tails() {
    let (config, x, y) = student_t_fixture(3.0);
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
fn the_log_likelihood_is_the_t_density() {
    let df = 4.0;
    let (config, x, y) = student_t_fixture(df);
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    let by_draw = fitted.log_likelihood(&x, &y).unwrap();
    let per_draw = fitted.predict_draws(&x).unwrap();
    let sigmas = fitted.sigma();
    let ln_c = libm::lgamma(0.5 * (df + 1.0))
        - libm::lgamma(0.5 * df)
        - 0.5 * (df * std::f64::consts::PI).ln();
    for (d, draw) in by_draw.iter().enumerate() {
        for (i, &value) in draw.iter().enumerate() {
            let z_sq = (y[i] - per_draw[d][i]).powi(2) / (df * sigmas[d] * sigmas[d]);
            let expected = ln_c - sigmas[d].ln() - 0.5 * (df + 1.0) * (1.0 + z_sq).ln();
            assert!((value - expected).abs() < 1e-9, "{value} vs {expected}");
        }
    }
}

#[test]
fn predict_variance_is_the_error_variance() {
    let df = 5.0;
    let (config, x, y) = student_t_fixture(df);
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    let variances = fitted.predict_variance(&x).unwrap();
    for (draw, sigma) in variances.iter().zip(fitted.sigma()) {
        let expected = sigma * sigma * df / (df - 2.0);
        assert!(draw.iter().all(|v| (v - expected).abs() < 1e-9 * expected));
    }
    let heavy = fit(&config.with_outcome(Outcome::student_t(2.0)), &x, &y, SEED).unwrap();
    assert!(matches!(
        heavy.predict_variance(&x),
        Err(Error::NotApplicable { .. })
    ));
    assert!(heavy.prediction_interval(&x, 0.9).is_ok());
}

#[test]
fn a_grid_stores_one_df_per_draw_and_round_trips() {
    let (config, x, y) = student_t_fixture(4.0);
    let grid = vec![3.0, 6.0, 12.0, 24.0];
    let config = config.with_outcome(Outcome::student_t_grid(grid.clone()));
    let fitted = fit(&config, &x, &y, SEED).unwrap();
    let dfs = fitted.posterior().dfs();
    assert_eq!(dfs.len(), fitted.n_draws());
    assert!(dfs.iter().all(|df| grid.contains(df)));
    let json = serde_json::to_string(&fitted).unwrap();
    let back: Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(back.posterior().dfs(), dfs);
    assert_eq!(
        fitted.log_likelihood(&x, &y).unwrap(),
        back.log_likelihood(&x, &y).unwrap()
    );
    // A grid reaching 2 admits a t without a variance.
    let wide = fit(
        &config.with_outcome(Outcome::student_t_grid(vec![2.0, 4.0])),
        &x,
        &y,
        SEED,
    )
    .unwrap();
    assert!(matches!(
        wide.predict_variance(&x),
        Err(Error::NotApplicable { .. })
    ));
}

#[test]
fn a_variance_ensemble_is_rejected_for_identification() {
    let (config, x, y) = student_t_fixture(4.0);
    let err = fit(&config.with_m_var(5), &x, &y, SEED).unwrap_err();
    assert!(err.to_string().contains("identification"), "{err}");
}

#[test]
fn the_degrees_of_freedom_are_validated() {
    let (config, x, y) = fixture();
    let cases: [(Outcome, &str); 5] = [
        (Outcome::student_t(0.0), "finite and positive"),
        (Outcome::student_t(f64::NAN), "finite and positive"),
        (Outcome::student_t_grid(vec![4.0]), "at least two"),
        (
            Outcome::student_t_grid(vec![5.0, 3.0]),
            "strictly increasing",
        ),
        (
            Outcome::student_t_grid(vec![3.0, f64::INFINITY]),
            "finite and positive",
        ),
    ];
    for (outcome, message) in cases {
        let err = fit(&config.clone().with_outcome(outcome), &x, &y, SEED).unwrap_err();
        assert!(err.to_string().contains(message), "{err}");
    }
}

#[test]
fn the_outcome_serialises_in_snake_case_and_round_trips() {
    let (config, _, _) = student_t_fixture(4.0);
    let json = serde_json::to_string(&config).unwrap();
    assert!(
        json.contains(r#""outcome":{"student_t":{"df":4.0,"nu":6.0,"q":0.85}}"#),
        "{json}"
    );
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(config, back);

    let grid = config.with_outcome(Outcome::student_t_grid(vec![3.0, 6.0]));
    let json = serde_json::to_string(&grid).unwrap();
    assert!(
        json.contains(r#""outcome":{"student_t":{"df":[3.0,6.0],"nu":6.0,"q":0.85}}"#),
        "{json}"
    );
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(grid, back);

    // An integer df reads as the fixed form.
    let integer: Config = serde_json::from_str(r#"{"outcome": {"student_t": {"df": 4}}}"#).unwrap();
    assert_eq!(integer.outcome, Outcome::student_t(4.0));
}
