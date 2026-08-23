//! Informational comparison against the authors' research scripts for the
//! probit and heteroscedastic variants. The scripts carry the
//! structural-move terms that CRAN AddiVortes corrected in 0.6.8, so they
//! target a different posterior from the crate; the tests print a
//! comparison table and write the crate's summaries, and assert nothing
//! beyond the presence of the script's output. They are ignored by
//! default and run by hand after the R script under `benchmarks/upstream/`:
//!
//! ```text
//! (cd benchmarks/upstream && Rscript binary_variant.R)
//! (cd benchmarks/upstream && Rscript heteroscedastic_variant.R)
//! (cd benchmarks/upstream && Rscript aft_abart.R)
//! cargo test --release --features experimental --test variants -- --ignored --nocapture
//! ```
//!
//! The AFT comparison is against BART `abart`, a different prior (trees
//! against tessellations), so its posteriors are close but not equal;
//! the report is the record.

use thiessen::{fit, Config, Data, Outcome};

const DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/variants");

/// Covariate columns then y, one header line.
fn parse_data(path: &str) -> (Data, Vec<f64>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
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
fn parse_summary(path: &str) -> Vec<(String, f64, f64)> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{path}: {e}"))
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

/// Covariate columns, then time and delta, one header line.
#[cfg(feature = "experimental")]
fn parse_survival_data(path: &str) -> (Data, Vec<f64>, Vec<bool>) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut lines = text.lines();
    let p = lines.next().unwrap().split(',').count() - 2;
    let mut values = Vec::new();
    let mut times = Vec::new();
    let mut events = Vec::new();
    for line in lines {
        let mut fields = line.split(',').map(|v| v.parse::<f64>().unwrap());
        for _ in 0..p {
            values.push(fields.next().unwrap());
        }
        times.push(fields.next().unwrap());
        events.push(fields.next().unwrap() == 1.0);
    }
    let n = times.len();
    (Data::new(values, n, p).unwrap(), times, events)
}

fn mean(series: &[f64]) -> f64 {
    series.iter().sum::<f64>() / series.len() as f64
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

fn report(name: &str, script: &[(String, f64, f64)], ours: &[(String, f64, f64)]) {
    let mut lines = vec!["summary,value,mcse".to_string()];
    println!("{name}: summary, script, crate, difference over combined MCSE");
    for ((label, theirs, mcse_theirs), (_, v, mcse_ours)) in script.iter().zip(ours) {
        let combined = (mcse_theirs * mcse_theirs + mcse_ours * mcse_ours).sqrt();
        println!(
            "  {label}: {theirs:.4} ({mcse_theirs:.4}), {v:.4} ({mcse_ours:.4}), {:.2}",
            (v - theirs) / combined
        );
        lines.push(format!("{label},{v},{mcse_ours}"));
    }
    std::fs::write(
        format!("{DIR}/{name}_core_summary.csv"),
        lines.join("\n") + "\n",
    )
    .unwrap();
}

/// Both sides: m = 50, 200 burn-in sweeps, 1000 kept draws, k = 3,
/// sigma_c = 0.8, omega = 3, lambda_c = 5. The script has no offset, so
/// the crate runs with offset 0 to match.
#[test]
#[ignore = "informational; needs the output of benchmarks/upstream/binary_variant.R"]
fn probit_friedman_against_the_binary_script() {
    let (x, y) = parse_data(&format!("{DIR}/probit_friedman_data.csv"));
    let script = parse_summary(&format!("{DIR}/probit_friedman_script_summary.csv"));
    let config = Config::new()
        .with_outcome(Outcome::probit())
        .with_offset(0.0)
        .with_m(50)
        .with_burn_in(200)
        .with_draws(1000);
    let fitted = fit(&config, &x, &y, 11).unwrap();
    let points = [0usize, 49, 99, 149, 199];
    let rows: Vec<&[f64]> = points.iter().map(|&r| x.row(r)).collect();
    let draws = fitted
        .predict_draws(&Data::from_rows(&rows).unwrap())
        .unwrap();
    let ours: Vec<(String, f64, f64)> = points
        .iter()
        .enumerate()
        .map(|(i, &row)| {
            let series: Vec<f64> = draws.iter().map(|d| d[i]).collect();
            (format!("p_mean_r{row}"), mean(&series), mcse_mean(&series))
        })
        .collect();
    assert_eq!(script.len(), ours.len());
    assert!(ours.iter().all(|(_, v, _)| v.is_finite()));
    report("probit_friedman", &script, &ours);
}

/// Both sides: m = 50, m' = 20, 200 burn-in sweeps, 1000 kept draws,
/// nu = 6, q = 0.85, k = 3, sigma_c = 0.8, omega = 3, lambda_c = 5.
#[test]
#[ignore = "informational; needs the output of benchmarks/upstream/heteroscedastic_variant.R"]
fn heteroscedastic_friedman_against_the_script() {
    let (x, y) = parse_data(&format!("{DIR}/heteroscedastic_friedman_data.csv"));
    let script = parse_summary(&format!(
        "{DIR}/heteroscedastic_friedman_script_summary.csv"
    ));
    let config = Config::new()
        .with_m(50)
        .with_m_var(20)
        .with_burn_in(200)
        .with_draws(1000);
    let fitted = fit(&config, &x, &y, 11).unwrap();
    let points = [0usize, 49, 99, 149, 199];
    let rows: Vec<&[f64]> = points.iter().map(|&r| x.row(r)).collect();
    let test = Data::from_rows(&rows).unwrap();
    let f_draws = fitted.predict_draws(&test).unwrap();
    let s2_draws = fitted.predict_variance(&test).unwrap();
    let mut ours: Vec<(String, f64, f64)> = Vec::new();
    for (prefix, draws) in [("f_mean", &f_draws), ("s2_mean", &s2_draws)] {
        for (i, &row) in points.iter().enumerate() {
            let series: Vec<f64> = draws.iter().map(|d| d[i]).collect();
            ours.push((
                format!("{prefix}_r{row}"),
                mean(&series),
                mcse_mean(&series),
            ));
        }
    }
    assert_eq!(script.len(), ours.len());
    assert!(ours.iter().all(|(_, v, _)| v.is_finite()));
    report("heteroscedastic_friedman", &script, &ours);
}

/// Both sides: m = 50, 200 burn-in sweeps, 1000 kept draws, k = 3,
/// sigma_c = 0.8, omega = 3, lambda_c = 5; the script sets abart to the
/// same tree count and k. The sigma^2 priors differ where the surfaces
/// do (sigdf = 3, sigquant = 0.90 against nu = 6, q = 0.85).
#[cfg(feature = "experimental")]
#[test]
#[ignore = "informational; needs the output of benchmarks/upstream/aft_abart.R"]
fn aft_friedman_against_abart() {
    let (x, times, events) = parse_survival_data(&format!("{DIR}/aft_friedman_data.csv"));
    let script = parse_summary(&format!("{DIR}/aft_friedman_script_summary.csv"));
    let config = Config::new()
        .with_outcome(Outcome::aft())
        .with_m(50)
        .with_burn_in(200)
        .with_draws(1000);
    let fitted = thiessen::fit_aft(&config, &x, &times, &events, 11).unwrap();
    let points = [0usize, 49, 99, 149, 199];
    let rows: Vec<&[f64]> = points.iter().map(|&r| x.row(r)).collect();
    let draws = fitted
        .predict_draws(&Data::from_rows(&rows).unwrap())
        .unwrap();
    let mut ours: Vec<(String, f64, f64)> = points
        .iter()
        .enumerate()
        .map(|(i, &row)| {
            let series: Vec<f64> = draws.iter().map(|d| d[i]).collect();
            (format!("f_mean_r{row}"), mean(&series), mcse_mean(&series))
        })
        .collect();
    let sigma = fitted.sigma();
    ours.push(("sigma_mean".into(), mean(&sigma), mcse_mean(&sigma)));
    assert_eq!(script.len(), ours.len());
    assert!(ours.iter().all(|(_, v, _)| v.is_finite()));
    report("aft_friedman", &script, &ours);
}
