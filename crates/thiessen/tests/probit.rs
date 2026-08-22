//! The probit model through the public surface: the input contract,
//! prediction semantics, and recovery of a known generating process.

mod common;
use common::TestRng;
use thiessen::{fit, Config, Data, Error, Outcome};

fn normal_cdf(z: f64) -> f64 {
    0.5 * libm::erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

fn config() -> Config {
    Config::new()
        .with_outcome(Outcome::probit())
        .with_m(5)
        .with_burn_in(10)
        .with_draws(10)
}

#[test]
fn labels_are_validated_at_the_boundary() {
    let x = Data::from_rows(&[[1.0, 9.0], [2.0, 7.0], [3.0, 8.0], [4.0, 5.0]]).unwrap();
    assert_eq!(
        fit(&config(), &x, &[0.0, 1.0, 2.0, 1.0], 1).unwrap_err(),
        Error::InvalidLabel { row: 2 }
    );
    assert_eq!(
        fit(&config(), &x, &[1.0, 1.0, 1.0, 1.0], 1).unwrap_err(),
        Error::DegenerateResponse
    );
    assert!(matches!(
        fit(&config().with_offset(f64::INFINITY), &x, &[0.0, 1.0, 1.0, 0.0], 1).unwrap_err(),
        Error::InvalidHyperparameter { ref name, .. } if name == "offset"
    ));
    let fitted = fit(&config(), &x, &[0.0, 1.0, 1.0, 0.0], 1).unwrap();
    assert_eq!(fitted.model_name(), "probit");
    // The default offset is Phi^-1(ybar) = 0 at ybar = 1/2.
    assert!(fitted.config().offset().unwrap().abs() < 1e-9);
    let explicit = fit(&config().with_offset(0.3), &x, &[0.0, 1.0, 1.0, 0.0], 1).unwrap();
    assert_eq!(explicit.config().offset(), Some(0.3));
}

/// A probit Friedman function: P(y = 1 | x) = Phi(f(x)) with f the
/// centred Friedman (1991) function scaled to unit standard deviation.
/// The posterior-mean probabilities have a Brier score near the Bayes
/// score E[p (1 - p)] and are calibrated in the large.
#[test]
fn recovers_a_probit_friedman_function() {
    let n = 400;
    let p = 5;
    let mut rng = TestRng(2026);
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
    let truth: Vec<f64> = raw.iter().map(|v| normal_cdf((v - mean) / sd)).collect();
    let labels: Vec<f64> = truth
        .iter()
        .map(|&pr| f64::from(rng.uniform() < pr))
        .collect();
    let x = Data::from_rows(&rows).unwrap();

    let config = Config::new()
        .with_outcome(Outcome::probit())
        .with_m(50)
        .with_burn_in(300)
        .with_draws(300);
    let fitted = fit(&config, &x, &labels, 17).unwrap();
    let probs = fitted.predict(&x).unwrap();

    let brier = probs
        .iter()
        .zip(&labels)
        .map(|(pr, y)| (pr - y) * (pr - y))
        .sum::<f64>()
        / n as f64;
    let bayes = truth.iter().map(|pr| pr * (1.0 - pr)).sum::<f64>() / n as f64;
    // Tolerance 0.03 on the Brier score: the excess over the Bayes score
    // is the mean squared error of the probabilities.
    assert!(brier < bayes + 0.03, "Brier {brier} against Bayes {bayes}");
    assert!(
        (fitted.in_sample_rmse() - brier.sqrt()).abs() < 1e-12,
        "in_sample_rmse is the root Brier score"
    );
    let rmse = (probs
        .iter()
        .zip(&truth)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / n as f64)
        .sqrt();
    assert!(rmse < 0.15, "probability rmse {rmse}");
    // Calibration in the large: mean predicted probability against the
    // share of ones, within 4 binomial standard errors.
    let share = labels.iter().sum::<f64>() / n as f64;
    let mean_prob = probs.iter().sum::<f64>() / n as f64;
    let se = (share * (1.0 - share) / n as f64).sqrt();
    assert!(
        (mean_prob - share).abs() < 4.0 * se,
        "{mean_prob} vs {share} +- {se}"
    );
    // Probabilities are monotone in the latent mean, so the credible
    // interval of the probability is inside [0, 1] and contains the mean.
    let ci = fitted.credible_interval(&x, 0.9).unwrap();
    for (c, pr) in ci.iter().zip(&probs) {
        assert!(0.0 <= c.lower && c.lower <= *pr && *pr <= c.upper && c.upper <= 1.0);
    }
}
