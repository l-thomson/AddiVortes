//! Prior-only sampling: the likelihood is off, so draws follow the prior
//! that a fit on the same data would use.

use thiessen::{fit, Config, Data};

/// Two responses sharing min and max, so both fix the same scaling, with
/// different interiors. Every value is exact rational arithmetic.
fn fixture() -> (Data, Vec<f64>, Vec<f64>) {
    let n = 40;
    let xs: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let interior = |i: usize, phase: usize| -> f64 {
        if i == 0 {
            0.0
        } else if i == n - 1 {
            1.0
        } else {
            0.2 + 0.6 * (((i * 13 + phase) % 23) as f64 / 22.0)
        }
    };
    let y1: Vec<f64> = (0..n).map(|i| interior(i, 0)).collect();
    let y2: Vec<f64> = (0..n).map(|i| interior(i, 7)).collect();
    (Data::new(xs, n, 1).unwrap(), y1, y2)
}

fn config() -> Config {
    Config::new()
        .with_prior_only(true)
        .with_m(1)
        .with_burn_in(0)
        .with_draws(4000)
}

/// Residual standard deviation of the least-squares line through the
/// scaled data, the sigma_hat of the prior calibration (both x and y have
/// training range 1 here, so scaling is a shift and the residuals are
/// unchanged).
fn sigma_hat(x: &Data, y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let xs: Vec<f64> = x.values().to_vec();
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let sxx: f64 = xs.iter().map(|v| (v - mean_x) * (v - mean_x)).sum();
    let sxy: f64 = xs
        .iter()
        .zip(y)
        .map(|(v, w)| (v - mean_x) * (w - mean_y))
        .sum();
    let beta = sxy / sxx;
    let rss: f64 = xs
        .iter()
        .zip(y)
        .map(|(v, w)| {
            let r = w - mean_y - beta * (v - mean_x);
            r * r
        })
        .sum();
    (rss / (n - 2.0)).sqrt()
}

#[test]
fn structure_and_means_ignore_the_response() {
    let (x, y1, y2) = fixture();
    let a = fit(&config(), &x, &y1, 11).unwrap();
    let b = fit(&config(), &x, &y2, 11).unwrap();
    assert_eq!(a.posterior().tessellations(), b.posterior().tessellations());
    assert_ne!(a.posterior().sigma_sq(), b.posterior().sigma_sq());
}

#[test]
fn sigma_prior_calibration_holds() {
    // lambda satisfies Pr(sigma < sigma_hat) = q, so the fraction of prior
    // sigma draws below sigma_hat is q within Monte Carlo error. The draws
    // are independent: the prior conditional does not depend on the state.
    let (x, y, _) = fixture();
    let fitted = fit(&config(), &x, &y, 3).unwrap();
    let threshold = sigma_hat(&x, &y);
    let sigma = fitted.sigma();
    let below = sigma.iter().filter(|s| **s < threshold).count() as f64;
    let frequency = below / sigma.len() as f64;
    let q = fitted.config().q;
    let tolerance = 4.0 * (q * (1.0 - q) / sigma.len() as f64).sqrt();
    assert!(
        (frequency - q).abs() < tolerance,
        "{frequency} vs {q} +- {tolerance}"
    );
}

#[test]
fn cell_means_have_the_prior_spread() {
    // mu ~ N(0, sigma_mu^2), sigma_mu = 0.5 / (k sqrt m) = 1/6 at k = 3,
    // m = 1, drawn fresh every sweep.
    let (x, y, _) = fixture();
    let fitted = fit(&config(), &x, &y, 5).unwrap();
    let mus: Vec<f64> = fitted
        .posterior()
        .tessellations()
        .iter()
        .flatten()
        .flat_map(|t| t.mus().iter().copied())
        .collect();
    assert!(mus.len() >= 4000);
    let n = mus.len() as f64;
    let mean = mus.iter().sum::<f64>() / n;
    let sd = (mus.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    let sigma_mu = 0.5 / 3.0;
    // Standard error of the sample sd of n normal draws: sigma_mu / sqrt(2n).
    let tolerance = 4.0 * sigma_mu / (2.0 * n).sqrt();
    assert!((sd - sigma_mu).abs() < tolerance, "{sd} vs {sigma_mu}");
    assert!(mean.abs() < 4.0 * sigma_mu / n.sqrt(), "{mean}");
}

#[test]
fn prior_predictive_path_and_serde() {
    let (x, y, _) = fixture();
    let fitted = fit(&config().with_draws(50), &x, &y, 9).unwrap();
    let predictions = fitted.predict(&x).unwrap();
    assert_eq!(predictions.len(), x.n_rows());
    assert!(predictions.iter().all(|p| p.is_finite()));
    assert!(fitted.prediction_interval(&x, 0.9).is_ok());

    assert!(!Config::default().prior_only);
    let parsed: Config = serde_json::from_str(r#"{"m": 4}"#).unwrap();
    assert!(!parsed.prior_only);
    let parsed: Config = serde_json::from_str(r#"{"prior_only": true}"#).unwrap();
    assert!(parsed.prior_only);
    let back: Config = serde_json::from_str(&serde_json::to_string(&parsed).unwrap()).unwrap();
    assert_eq!(back, parsed);
}
