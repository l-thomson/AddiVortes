//! The heteroscedastic model through the public surface: the input
//! contract, prediction semantics, the prior of s^2, and recovery of a
//! known generating process.

mod common;
use common::TestRng;
use thiessen::{fit, Config, Data, Error, Model, Sampler};

fn config() -> Config {
    Config::new()
        .with_model(Model::Heteroscedastic)
        .with_m(5)
        .with_m_var(3)
        .with_burn_in(10)
        .with_draws(12)
}

fn toy() -> (Data, Vec<f64>) {
    let rows: Vec<[f64; 2]> = (0..20)
        .map(|i| [i as f64 / 19.0, ((i * 7) % 20) as f64 / 20.0])
        .collect();
    let y = rows
        .iter()
        .enumerate()
        .map(|(i, r)| r[0] + 0.5 * r[1] + 0.2 * (((i * 11) % 13) as f64 / 12.0 - 0.5))
        .collect();
    (Data::from_rows(&rows).unwrap(), y)
}

#[test]
fn hyperparameters_are_validated_at_the_boundary() {
    let (x, y) = toy();
    assert!(matches!(
        fit(&config().with_m_var(0), &x, &y, 1).unwrap_err(),
        Error::InvalidHyperparameter { ref name, .. } if name == "m_var"
    ));
    assert!(matches!(
        fit(&config().with_nu(2.0), &x, &y, 1).unwrap_err(),
        Error::InvalidHyperparameter { ref name, .. } if name == "nu"
    ));
    assert!(Config::new().with_m_var(0).validate().is_ok());
}

#[test]
fn prediction_surface_under_the_model() {
    let (x, y) = toy();
    let fitted = fit(&config(), &x, &y, 3).unwrap();
    assert_eq!(fitted.model(), Model::Heteroscedastic);
    assert!(fitted.sigma().is_empty());
    assert!(fitted.posterior().sigma_sq().is_empty());
    assert_eq!(fitted.posterior().variance_tessellations().len(), 12);

    let variances = fitted.predict_variance(&x).unwrap();
    assert_eq!(variances.len(), 12);
    assert!(variances
        .iter()
        .all(|d| d.len() == 20 && d.iter().all(|v| v.is_finite() && *v > 0.0)));

    let mean = fitted.predict(&x).unwrap();
    for (interval, m) in fitted
        .prediction_interval(&x, 0.9)
        .unwrap()
        .iter()
        .zip(&mean)
    {
        assert!(interval.lower < *m && *m < interval.upper);
    }
    let ll = fitted.log_likelihood(&x, &y).unwrap();
    assert_eq!(ll.len(), 12);
    assert!(ll.iter().flatten().all(|v| v.is_finite()));

    let json = serde_json::to_string(&fitted).unwrap();
    let loaded: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded, fitted);
    assert_eq!(loaded.predict_variance(&x).unwrap(), variances);
}

/// Under the prior, E[s^2(x)] = nu lambda / (nu - 2) for every m', the
/// prior mean of the Gaussian model's sigma^2; checked by prior-only
/// sampling under the pinned prior within 4 batch-means standard errors.
#[test]
fn prior_mean_of_the_variance_product_is_matched() {
    let (x, y) = toy();
    let (nu, lambda) = (6.0, 0.04);
    let expected = nu * lambda / (nu - 2.0);
    for (m_var, seed) in [(1, 31_u64), (5, 32), (40, 33)] {
        let config = Config::new()
            .with_model(Model::Heteroscedastic)
            .with_m(2)
            .with_m_var(m_var)
            .with_nu(nu)
            .with_prior_only(true);
        let mut sampler = Sampler::pinned_prior(&config, &x, &y, lambda, seed).unwrap();
        let draws = 3000;
        let mut means = Vec::with_capacity(draws);
        for _ in 0..50 {
            sampler.step();
        }
        for _ in 0..draws {
            sampler.step();
            let v = sampler.noise_variances();
            means.push(v.iter().sum::<f64>() / v.len() as f64);
        }
        let batches = 30;
        let size = draws / batches;
        let batch_means: Vec<f64> = (0..batches)
            .map(|b| means[b * size..(b + 1) * size].iter().sum::<f64>() / size as f64)
            .collect();
        let overall = batch_means.iter().sum::<f64>() / batches as f64;
        let se = (batch_means
            .iter()
            .map(|m| (m - overall) * (m - overall))
            .sum::<f64>()
            / (batches as f64 - 1.0)
            / batches as f64)
            .sqrt();
        assert!(
            (overall - expected).abs() < 4.0 * se,
            "m' = {m_var}: mean {overall} against {expected}, se {se}"
        );
    }
}

/// A heteroscedastic Friedman function: y = f(x) + s(x) e with f the
/// centred Friedman (1991) function scaled to unit standard deviation and
/// s(x) = 0.3 + 0.7 x_1. The posterior mean recovers f and ln s^2 within
/// stated root-mean-square errors; the 90% prediction-interval coverage
/// is reported, not gated.
#[test]
fn recovers_a_heteroscedastic_friedman_function() {
    let n = 400;
    let p = 5;
    let mut rng = TestRng(2027);
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..p).map(|_| rng.uniform()).collect())
        .collect();
    let raw: Vec<f64> = rows
        .iter()
        .map(|r| {
            10.0 * (std::f64::consts::PI * r[0] * r[1]).sin()
                + 20.0 * (r[2] - 0.5) * (r[2] - 0.5)
                + 10.0 * r[3]
                + 5.0 * r[4]
        })
        .collect();
    let mean = raw.iter().sum::<f64>() / n as f64;
    let sd = (raw.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
    let f: Vec<f64> = raw.iter().map(|v| (v - mean) / sd).collect();
    let s: Vec<f64> = rows.iter().map(|r| 0.3 + 0.7 * r[0]).collect();
    let y: Vec<f64> = f
        .iter()
        .zip(&s)
        .map(|(f, s)| f + s * rng.normal())
        .collect();
    let x = Data::from_rows(&rows).unwrap();

    let config = Config::new()
        .with_model(Model::Heteroscedastic)
        .with_m(50)
        .with_m_var(20)
        .with_burn_in(300)
        .with_draws(300);
    let fitted = fit(&config, &x, &y, 19).unwrap();
    let f_hat = fitted.predict(&x).unwrap();
    let variances = fitted.predict_variance(&x).unwrap();
    let log_s2_hat: Vec<f64> = (0..n)
        .map(|i| variances.iter().map(|d| d[i].ln()).sum::<f64>() / variances.len() as f64)
        .collect();

    let rmse = |a: &[f64], b: &[f64]| {
        (a.iter().zip(b).map(|(u, v)| (u - v) * (u - v)).sum::<f64>() / n as f64).sqrt()
    };
    let rmse_f = rmse(&f_hat, &f);
    let log_s2: Vec<f64> = s.iter().map(|v| 2.0 * v.ln()).collect();
    let rmse_log_s2 = rmse(&log_s2_hat, &log_s2);
    // Tolerances are loose by design: 0.35 on f (noise sd 0.3 to 1.0 on a
    // unit-sd signal) and 0.6 on ln s^2 (a factor 1.8 on s^2).
    assert!(rmse_f < 0.35, "rmse of f {rmse_f}");
    assert!(rmse_log_s2 < 0.6, "rmse of ln s^2 {rmse_log_s2}");

    let intervals = fitted.prediction_interval(&x, 0.9).unwrap();
    let coverage = intervals
        .iter()
        .zip(&y)
        .filter(|(i, y)| i.lower <= **y && **y <= i.upper)
        .count() as f64
        / n as f64;
    let bias = log_s2_hat
        .iter()
        .zip(&log_s2)
        .map(|(a, b)| a - b)
        .sum::<f64>()
        / n as f64;
    println!(
        "rmse f {rmse_f:.3}, rmse ln s^2 {rmse_log_s2:.3}, bias ln s^2 {bias:.3}, 90% coverage {coverage:.3}"
    );
}
