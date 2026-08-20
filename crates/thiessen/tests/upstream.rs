//! Comparison against the upstream R package (CRAN AddiVortes 0.6.9,
//! pinned in benchmarks/upstream/renv.lock): posterior summaries on fixed
//! datasets agree within k = 4 combined Monte Carlo standard errors.
//! Fixtures under tests/fixtures/upstream are written only by
//! benchmarks/upstream/generate.R.
//!
//! k = 4 is a two-sided 6e-5 level per summary, which covers the summary
//! set at a family-wise level under 1 percent with no further
//! correction.

use thiessen::{fit, Config, Data};

const K: f64 = 4.0;
/// Half-width of the central difference estimating the density at a
/// quantile; matches the fixture script.
const H: f64 = 0.025;

fn fixture(name: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/upstream/{name}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"))
}

/// Covariate columns then y, one header line, as written by the script.
fn parse_data(name: &str) -> (Data, Vec<f64>) {
    let text = fixture(name);
    let mut lines = text.lines();
    let p = lines.next().unwrap().split(',').count() - 1;
    let mut values = Vec::new();
    let mut y = Vec::new();
    for line in lines {
        let mut fields = line.split(',').map(|v| v.parse::<f64>().unwrap());
        for _ in 0..p {
            values.push(fields.next().unwrap());
        }
        y.push(fields.next().unwrap());
    }
    let n = y.len();
    (Data::new(values, n, p).unwrap(), y)
}

/// Rows of (summary, value, mcse).
fn parse_summary(name: &str) -> Vec<(String, f64, f64)> {
    fixture(name)
        .lines()
        .skip(1)
        .map(|line| {
            let mut fields = line.split(',');
            (
                fields.next().unwrap().to_string(),
                fields.next().unwrap().parse().unwrap(),
                fields.next().unwrap().parse().unwrap(),
            )
        })
        .collect()
}

fn mean(series: &[f64]) -> f64 {
    series.iter().sum::<f64>() / series.len() as f64
}

fn sd(series: &[f64]) -> f64 {
    let m = mean(series);
    (series.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (series.len() as f64 - 1.0)).sqrt()
}

/// MCSE of the mean by batch means, 25 batches.
fn mcse_mean(series: &[f64]) -> f64 {
    let batches = 25;
    let size = series.len() / batches;
    let means: Vec<f64> = (0..batches)
        .map(|k| series[k * size..(k + 1) * size].iter().sum::<f64>() / size as f64)
        .collect();
    let centre = mean(&means);
    let var = means
        .iter()
        .map(|m| (m - centre) * (m - centre))
        .sum::<f64>()
        / (batches as f64 - 1.0);
    (var / batches as f64).sqrt()
}

/// Effective sample size implied by the batch-means MCSE, in [1, n].
fn ess(series: &[f64]) -> f64 {
    let mcse = mcse_mean(series);
    if mcse == 0.0 {
        return series.len() as f64;
    }
    let ratio = sd(series) / mcse;
    (ratio * ratio).clamp(1.0, series.len() as f64)
}

/// Type 7 quantile of an unsorted series.
fn quantile(series: &[f64], p: f64) -> f64 {
    let mut sorted = series.to_vec();
    sorted.sort_by(f64::total_cmp);
    let h = p * (sorted.len() - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo as f64)
}

/// MCSE of a quantile: sqrt(p (1 - p) / ESS) over the central-difference
/// density; matches the fixture script.
fn quantile_mcse(series: &[f64], p: f64) -> f64 {
    let width = (quantile(series, p + H) - quantile(series, p - H)).max(1e-12);
    (p * (1.0 - p) / ess(series)).sqrt() * width / (2.0 * H)
}

fn compare(dataset: &str, points: &[usize], seed: u64) {
    let (x, y) = parse_data(&format!("{dataset}_data.csv"));
    let upstream = parse_summary(&format!("{dataset}_summary.csv"));

    // The defaults match the fixture script's arguments: m = 200, nu = 6,
    // q = 0.85, k = 3, sigma_c = 0.8, omega = min(3, p), lambda_c = 5,
    // 200 burn-in sweeps, 1000 kept draws.
    let fitted = fit(&Config::new(), &x, &y, seed).unwrap();

    let rows: Vec<&[f64]> = points.iter().map(|&r| x.row(r)).collect();
    let x_points = Data::from_rows(&rows).unwrap();
    let draws = fitted.predict_draws(&x_points).unwrap();
    let sigma = fitted.sigma();

    let mut ours = std::collections::BTreeMap::new();
    for (i, &row) in points.iter().enumerate() {
        let series: Vec<f64> = draws.iter().map(|d| d[i]).collect();
        let e = ess(&series);
        let s = sd(&series);
        ours.insert(format!("f_mean_r{row}"), (mean(&series), s / e.sqrt()));
        ours.insert(format!("f_sd_r{row}"), (s, s / (2.0 * e).sqrt()));
    }
    let e = ess(&sigma);
    let s = sd(&sigma);
    ours.insert("sigma_mean".to_string(), (mean(&sigma), s / e.sqrt()));
    for p in [0.05_f64, 0.5, 0.95] {
        ours.insert(
            format!("sigma_q{:02}", (100.0 * p).round() as u32),
            (quantile(&sigma, p), quantile_mcse(&sigma, p)),
        );
    }

    assert_eq!(upstream.len(), ours.len(), "{dataset}: summary sets differ");
    for (name, value, mcse_upstream) in upstream {
        let (v, mcse_ours) = ours[&name];
        let tolerance = K * (mcse_upstream * mcse_upstream + mcse_ours * mcse_ours).sqrt();
        assert!(
            (v - value).abs() < tolerance,
            "{dataset} {name}: {v} vs upstream {value}, tolerance {tolerance}"
        );
    }
}

#[test]
fn friedman_matches_upstream() {
    compare("friedman", &[0, 49, 99, 149, 199], 11);
}

#[test]
fn attitude_matches_upstream() {
    compare("attitude", &[0, 7, 14, 21, 29], 12);
}
