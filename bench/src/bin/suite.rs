//! One chain of one suite cell, in its own process.
//!
//! The scorecard is computed in `benchmarks/suite`, by a pinned ArviZ over
//! the draws this writes. Nothing here estimates an effective sample size
//! or a potential scale reduction: ESS estimators disagree materially on
//! poorly mixed chains, so the estimator is part of the measurement
//! definition and there is exactly one of it.
//!
//! One process per chain gives an attributable peak resident set, and a
//! run resumes from whatever it has already written.
//!
//! ```text
//! suite list
//! suite run <model> <n> <p> <seed> <chain> <out-dir>
//! ```

use std::io::Write;
use std::time::Instant;

use thiessen::{chain_seed, Fitted};
use thiessen_bench::{build_cell, cells, Cell, CELL_BURN_IN, CELL_DRAWS, DECLARED_ROWS, HOLDOUT};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("list") => list(),
        Some("run") if args.len() == 7 => run(&args[1..]),
        _ => {
            eprintln!("usage: suite list");
            eprintln!("       suite run <model> <n> <p> <seed> <chain> <out-dir>");
            std::process::exit(2);
        }
    }
}

/// The cell set as JSON, so the driver enumerates the registry rather than
/// carrying its own copy of it.
fn list() {
    let cells: Vec<serde_json::Value> = cells()
        .iter()
        .map(|cell| {
            serde_json::json!({
                "id": cell.id(),
                "model": cell.model,
                "n": cell.n,
                "p": cell.p,
            })
        })
        .collect();
    let doc = serde_json::json!({
        "burn_in": CELL_BURN_IN,
        "draws": CELL_DRAWS,
        "holdout": HOLDOUT,
        "declared_rows": DECLARED_ROWS,
        "cells": cells,
    });
    println!("{}", serde_json::to_string_pretty(&doc).unwrap());
}

fn run(args: &[String]) {
    let model = args[0].clone();
    let n: usize = args[1].parse().expect("n is a whole number");
    let p: usize = args[2].parse().expect("p is a whole number");
    let seed: u64 = args[3].parse().expect("seed is a whole number");
    let chain: usize = args[4].parse().expect("chain is a whole number");
    let out = std::path::PathBuf::from(&args[5]);

    let cell = Cell {
        model: leak(model),
        n,
        p,
    };
    let split = build_cell(&cell);
    let sweeps = CELL_BURN_IN + CELL_DRAWS;

    // The schedule is driven here rather than through `fit` so that the
    // post-warm-up time is measured on its own for every model: burn-in
    // and initialisation are not part of the per-iteration cost a
    // published efficiency number quotes (the bartz method,
    // arXiv:2410.23244 C.1). The order of sweeps and keeps is `fit`'s at
    // thinning 1, so the draws are the draws `fit` would give.
    let start = Instant::now();
    let mut sampler = split.train.sampler(chain_seed(seed, chain));
    for _ in 0..CELL_BURN_IN {
        sampler.step();
    }
    let warmup = start.elapsed();
    for _ in 0..CELL_DRAWS {
        sampler.step();
        sampler.keep();
    }
    let fit = start.elapsed();
    let fitted = sampler
        .finish()
        .expect("the schedule keeps at least one draw");

    let predict_start = Instant::now();
    let predictions = fitted.predict(&split.test_x).expect("held-out predict");
    let predict = predict_start.elapsed();

    std::fs::create_dir_all(&out).expect("output directory");
    let stem = format!("{}-chain{chain}", cell.id());
    write_draws(&fitted, &split, chain, &out.join(format!("{stem}.csv")));

    let accuracy = split
        .test_y
        .as_ref()
        .map(|y| accuracy(&fitted, &split, y, &predictions));
    let doc = serde_json::json!({
        "cell": cell.id(),
        "model": cell.model,
        "n": cell.n,
        "p": cell.p,
        "seed": seed,
        "chain": chain,
        "chain_seed": chain_seed(seed, chain),
        "sweeps": sweeps,
        "burn_in": CELL_BURN_IN,
        "draws": CELL_DRAWS,
        "fit_seconds": fit.as_secs_f64(),
        "warmup_seconds": warmup.as_secs_f64(),
        "post_warmup_seconds": (fit - warmup).as_secs_f64(),
        "predict_seconds": predict.as_secs_f64(),
        "accuracy": accuracy,
        "core_version": thiessen::VERSION,
    });
    std::fs::write(
        out.join(format!("{stem}.json")),
        serde_json::to_string_pretty(&doc).unwrap(),
    )
    .expect("run metadata");
}

/// The declared quantities, one row per draw: sigma where the model
/// samples one, f(x) at the first held-out rows, and the two structure
/// counts.
fn write_draws(
    fitted: &Fitted,
    split: &thiessen_bench::Split,
    chain: usize,
    path: &std::path::Path,
) {
    let file = std::fs::File::create(path).expect("draws file");
    let mut out = std::io::BufWriter::new(file);
    writeln!(out, "chain,draw,quantity,value").unwrap();

    let series = |name: &str, values: &[f64], out: &mut dyn Write| {
        for (draw, value) in values.iter().enumerate() {
            writeln!(out, "{chain},{draw},{name},{value:.17e}").unwrap();
        }
    };
    let sigma = fitted.sigma();
    if !sigma.is_empty() {
        series("sigma", &sigma, &mut out);
    }
    series("cells", &fitted.cell_counts(), &mut out);
    series("dims", &fitted.dimension_counts(), &mut out);

    let declared = DECLARED_ROWS.min(split.test_x.n_rows());
    let per_draw = fitted.predict_draws(&split.test_x).expect("held-out draws");
    for row in 0..declared {
        let values: Vec<f64> = per_draw.iter().map(|draw| draw[row]).collect();
        series(&format!("f[{row}]"), &values, &mut out);
    }
    out.flush().unwrap();
}

/// Held-out accuracy: root mean squared error of the posterior mean, mean
/// log predictive density, and the coverage and mean width of the 95 per
/// cent predictive interval.
fn accuracy(
    fitted: &Fitted,
    split: &thiessen_bench::Split,
    y: &[f64],
    predictions: &[f64],
) -> serde_json::Value {
    let n = y.len() as f64;
    let rmse = (predictions
        .iter()
        .zip(y)
        .map(|(p, t)| (p - t) * (p - t))
        .sum::<f64>()
        / n)
        .sqrt();

    // log (1/S sum_s p(y | theta_s)) per row, by the log-sum-exp of the
    // per-draw log densities.
    let lpd = fitted
        .log_likelihood(&split.test_x, y)
        .map(|per_draw| {
            let mut total = 0.0;
            for row in 0..y.len() {
                let column: Vec<f64> = per_draw.iter().map(|d| d[row]).collect();
                total += log_mean_exp(&column);
            }
            total / n
        })
        .ok();

    let interval = fitted.prediction_interval(&split.test_x, 0.95).ok();
    let (coverage, width) = match &interval {
        Some(bounds) => {
            let covered = bounds
                .iter()
                .zip(y)
                .filter(|(b, t)| **t >= b.lower && **t <= b.upper)
                .count() as f64
                / n;
            let width = bounds.iter().map(|b| b.upper - b.lower).sum::<f64>() / n;
            (Some(covered), Some(width))
        }
        None => (None, None),
    };

    serde_json::json!({
        "rmse": rmse,
        "lpd": lpd,
        "coverage_95": coverage,
        "width_95": width,
    })
}

fn log_mean_exp(values: &[f64]) -> f64 {
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return max;
    }
    let sum: f64 = values.iter().map(|v| (v - max).exp()).sum();
    max + (sum / values.len() as f64).ln()
}

/// The registry holds `&'static str` names; the model comes in as an
/// argument, so it outlives the process either way.
fn leak(name: String) -> &'static str {
    Box::leak(name.into_boxed_str())
}
