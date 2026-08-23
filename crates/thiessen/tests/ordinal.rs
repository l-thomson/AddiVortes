//! The ordinal model at the fit boundary: draw-for-draw agreement with
//! the probit model at two categories, the cutpoint draws, the
//! prediction semantics and persistence.

#![cfg(feature = "experimental")]

mod common;

use common::{fixture, probit_fixture, SEED};
use thiessen::{fit, Config, Error, Outcome, Sampler};

/// The Gaussian fixture's response cut into four ordered categories at
/// its quartiles, so every category is present.
fn ordinal_fixture() -> (Config, thiessen::Data, Vec<f64>) {
    let (config, x, y) = fixture();
    let mut sorted = y.clone();
    sorted.sort_by(f64::total_cmp);
    let cuts = [
        sorted[y.len() / 4],
        sorted[y.len() / 2],
        sorted[3 * y.len() / 4],
    ];
    let labels = y
        .iter()
        .map(|&v| cuts.iter().filter(|&&c| v >= c).count() as f64)
        .collect();
    (config.with_outcome(Outcome::ordinal(4)), x, labels)
}

#[test]
fn two_category_data_reproduces_the_probit_chain() {
    let (probit_config, x, labels) = probit_fixture();
    let probit = fit(&probit_config, &x, &labels, SEED).unwrap();
    let config = Config::new()
        .with_m(15)
        .with_burn_in(50)
        .with_draws(60)
        .with_outcome(Outcome::ordinal(2));
    let ordinal = fit(&config, &x, &labels, SEED).unwrap();
    assert_eq!(ordinal.model_name(), "ordinal");
    assert!(ordinal.cutpoint_draws().is_empty());
    assert_eq!(
        probit.predict_draws(&x).unwrap(),
        ordinal.predict_draws(&x).unwrap()
    );
    assert_eq!(
        probit.predict_latent(&x).unwrap(),
        ordinal.predict_latent(&x).unwrap()
    );
}

#[test]
fn the_cutpoints_are_sampled_and_the_fit_round_trips() {
    let (config, x, labels) = ordinal_fixture();
    let fitted = fit(&config, &x, &labels, SEED).unwrap();
    let draws = fitted.cutpoint_draws();
    assert_eq!(draws.len(), 60);
    assert!(draws
        .iter()
        .all(|d| d.len() == 2 && d[1] > d[0] && d[0] > 0.0));
    // The chain moves: not every draw holds the same cutpoints.
    assert!(draws.iter().any(|d| d != &draws[0]));
    let json = serde_json::to_string(&fitted).unwrap();
    let back: thiessen::Fitted = serde_json::from_str(&json).unwrap();
    assert_eq!(fitted.predict(&x).unwrap(), back.predict(&x).unwrap());
    assert_eq!(fitted.cutpoint_draws(), back.cutpoint_draws());
    assert_eq!(back.model_name(), "ordinal");
}

#[test]
fn the_category_probabilities_sum_to_one_and_order_the_expected_category() {
    let (config, x, labels) = ordinal_fixture();
    let fitted = fit(&config, &x, &labels, SEED).unwrap();
    let probabilities = fitted.predict_category_probabilities(&x).unwrap();
    let expected = fitted.predict(&x).unwrap();
    for (row, e) in probabilities.iter().zip(&expected) {
        assert_eq!(row.len(), 4);
        assert!(row.iter().all(|p| (0.0..=1.0).contains(p)));
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        let mean: f64 = row.iter().enumerate().map(|(k, p)| k as f64 * p).sum();
        assert!((mean - e).abs() < 1e-9);
    }
    let gaussian = fit(
        &Config::new().with_m(15).with_burn_in(50).with_draws(60),
        &x,
        &labels,
        SEED,
    )
    .unwrap();
    assert!(matches!(
        gaussian.predict_category_probabilities(&x),
        Err(Error::NotApplicable { .. })
    ));
}

#[test]
fn the_log_likelihood_carries_the_ordinal_terms() {
    let (config, x, labels) = ordinal_fixture();
    let fitted = fit(&config, &x, &labels, SEED).unwrap();
    let by_draw = fitted.log_likelihood(&x, &labels).unwrap();
    assert!(by_draw.iter().flatten().all(|v| v.is_finite() && *v < 0.0));
    let mut bad = labels.clone();
    bad[3] = 4.0;
    assert!(matches!(
        fitted.log_likelihood(&x, &bad),
        Err(Error::InvalidOrdinalLabel {
            row: 3,
            categories: 4
        })
    ));
    assert!(matches!(
        fitted.predict_variance(&x),
        Err(Error::NotApplicable { .. })
    ));
    assert!(matches!(
        fitted.prediction_interval(&x, 0.9),
        Err(Error::NotApplicable { .. })
    ));
    assert!(fitted.sigma().is_empty());
}

#[test]
fn the_labels_are_validated_at_fit() {
    let (config, x, mut labels) = ordinal_fixture();
    labels[5] = 1.5;
    assert!(matches!(
        fit(&config, &x, &labels, SEED),
        Err(Error::InvalidOrdinalLabel {
            row: 5,
            categories: 4
        })
    ));
    labels[5] = f64::NAN;
    assert!(matches!(
        fit(&config, &x, &labels, SEED),
        Err(Error::NonFiniteResponse { row: 5 })
    ));
    // An empty category is permitted; its cutpoint gap follows the
    // prior.
    let (_, _, labels) = ordinal_fixture();
    let folded: Vec<f64> = labels.iter().map(|&v| v.min(2.0)).collect();
    let fitted = fit(&config, &x, &folded, SEED).unwrap();
    assert!(fitted.cutpoint_draws().iter().all(|d| d[1] > d[0]));
}

#[test]
fn a_variance_ensemble_is_rejected_for_identification() {
    let (config, x, labels) = ordinal_fixture();
    let err = fit(&config.with_m_var(5), &x, &labels, SEED).unwrap_err();
    assert!(err.to_string().contains("identification"), "{err}");
}

#[test]
fn the_response_seam_replaces_the_codes() {
    let (config, x, labels) = ordinal_fixture();
    let mut sampler = Sampler::new(&config, &x, &labels, SEED).unwrap();
    sampler.step();
    let folded: Vec<f64> = labels.iter().map(|&v| v.min(2.0)).collect();
    // A replacement may leave a category empty; only the codes are
    // validated.
    sampler.set_response(&folded).unwrap();
    sampler.step();
    assert!(!sampler.cutpoints().is_empty());
    let mut bad = labels;
    bad[0] = -1.0;
    assert!(matches!(
        sampler.set_response(&bad),
        Err(Error::InvalidOrdinalLabel {
            row: 0,
            categories: 4
        })
    ));
}

/// Effective sample size by the batch-means estimator:
/// n Var(x) / (m Var(batch means)) with batch size m.
fn effective_sample_size(values: &[f64]) -> f64 {
    let batches = 50;
    let size = values.len() / batches;
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var =
        values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (values.len() as f64 - 1.0);
    let batch_means: Vec<f64> = (0..batches)
        .map(|k| values[k * size..(k + 1) * size].iter().sum::<f64>() / size as f64)
        .collect();
    let bm_var = batch_means
        .iter()
        .map(|m| (m - mean) * (m - mean))
        .sum::<f64>()
        / (batches as f64 - 1.0);
    values.len() as f64 * var / (size as f64 * bm_var)
}

/// The Cowles (1996) mixing concern at a full-size fit: the blocked
/// collapsed move must keep the cutpoint chains usable as n grows.
#[test]
#[ignore = "full size, nightly"]
fn the_cutpoint_chains_mix_at_full_size() {
    let n = 300;
    let mut rng = common::TestRng(2024);
    let rows: Vec<[f64; 2]> = (0..n)
        .map(|i| {
            [
                i as f64 / (n - 1) as f64,
                ((i * 37) % n) as f64 / (n - 1) as f64,
            ]
        })
        .collect();
    let cuts = [0.5, 1.2];
    let labels: Vec<f64> = rows
        .iter()
        .map(|r| {
            let z = 1.5 * (r[0] - 0.5) + r[1] - 0.5 + rng.normal();
            let mut k = 0.0;
            if z > 0.0 {
                k = 1.0;
                for &g in &cuts {
                    if z > g {
                        k += 1.0;
                    }
                }
            }
            k
        })
        .collect();
    let x = thiessen::Data::from_rows(&rows).unwrap();
    let config = Config::new()
        .with_outcome(Outcome::ordinal(4))
        .with_m(50)
        .with_burn_in(500)
        .with_draws(2000);
    let fitted = fit(&config, &x, &labels, 5).unwrap();
    for (j, series) in (0..2)
        .map(|j| {
            fitted
                .cutpoint_draws()
                .iter()
                .map(|d| d[j])
                .collect::<Vec<f64>>()
        })
        .enumerate()
    {
        let ess = effective_sample_size(&series);
        assert!(ess > 100.0, "gamma_{} ESS {ess} of 2000 kept draws", j + 2);
    }
}

#[test]
fn the_outcome_serialises_in_snake_case_and_round_trips() {
    let (config, _, _) = ordinal_fixture();
    let json = serde_json::to_string(&config).unwrap();
    assert!(
        json.contains(r#""outcome":{"ordinal":{"categories":4,"offset":null,"cutpoint_sd":1.0}}"#),
        "{json}"
    );
    let back: Config = serde_json::from_str(&json).unwrap();
    assert_eq!(config, back);
}
