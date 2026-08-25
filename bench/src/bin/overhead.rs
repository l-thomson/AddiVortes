//! The core's time on the four cases the binding benchmarks run, as JSON.
//!
//! A ratio on a two-hundred-microsecond call carries no information, so
//! the binding tables report the absolute difference against these numbers
//! beside the ratio. The configuration, the sizes and the seed are the
//! registry's, so the core and each binding provably do the same work.
//!
//! ```text
//! overhead run [repetitions]
//! overhead designs <dir>
//! ```
//!
//! `run` times each case `repetitions` times, three by default, and
//! reports the median. `designs` writes the designs and responses the
//! cases use, so a binding benchmark reads the same numbers rather than
//! generating its own: the core's generator is a splitmix64 no other
//! language reproduces, and a comparison over different data is a
//! comparison of different work.
//!
//! Comparable only against a binding measured on the same machine in the
//! same session.

use std::hint::black_box;
use std::time::Instant;

use thiessen::{fit, fit_with_progress, Data};
use thiessen_bench::{friedman, BINDING_SIZES, BINDING_SWEEPS, CASES, PREDICT_ROWS, SEED};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("designs") if args.len() == 2 => designs(std::path::Path::new(&args[1])),
        Some("run") => run(args.get(1).map(|v| v.parse().expect("a whole number"))),
        _ => {
            eprintln!("usage: overhead run [repetitions]");
            eprintln!("       overhead designs <dir>");
            std::process::exit(2);
        }
    }
}

/// Write one CSV per size, covariate columns then the response, and one
/// per column count for the predict matrix.
fn designs(dir: &std::path::Path) {
    std::fs::create_dir_all(dir).expect("design directory");
    let gaussian = CASES
        .iter()
        .find(|c| c.name == "gaussian")
        .expect("the registry carries the Gaussian model");
    for &(n, p) in BINDING_SIZES {
        let workload = (gaussian.build)(n, p);
        write_csv(
            &dir.join(format!("train-n{n}-p{p}.csv")),
            &workload.x,
            Some(workload.numeric()),
        );
        write_csv(
            &dir.join(format!("predict-p{p}.csv")),
            &predict_design(p),
            None,
        );
    }
}

fn write_csv(path: &std::path::Path, x: &Data, y: Option<&[f64]>) {
    use std::io::Write;
    let file = std::fs::File::create(path).expect("design file");
    let mut out = std::io::BufWriter::new(file);
    let header: Vec<String> = (1..=x.n_cols()).map(|i| format!("x{i}")).collect();
    write!(out, "{}", header.join(",")).unwrap();
    if y.is_some() {
        write!(out, ",y").unwrap();
    }
    writeln!(out).unwrap();
    for row in 0..x.n_rows() {
        let values: Vec<String> = x.row(row).iter().map(|v| format!("{v:.17e}")).collect();
        write!(out, "{}", values.join(",")).unwrap();
        if let Some(y) = y {
            write!(out, ",{:.17e}", y[row]).unwrap();
        }
        writeln!(out).unwrap();
    }
    out.flush().unwrap();
}

fn run(repetitions: Option<usize>) {
    let repetitions = repetitions.unwrap_or(3);

    let gaussian = CASES
        .iter()
        .find(|c| c.name == "gaussian")
        .expect("the registry carries the Gaussian model");

    let mut cases = Vec::new();
    for &(n, p) in BINDING_SIZES {
        let workload = (gaussian.build)(n, p);
        let y = workload.numeric().to_vec();

        cases.push(record("fit", n, p, repetitions, || {
            black_box(fit(&workload.config, &workload.x, &y, 1).unwrap());
        }));

        let large = predict_design(p);
        let fitted = workload.fit(1);
        cases.push(record("predict", n, p, repetitions, || {
            black_box(fitted.predict(&large).unwrap());
        }));

        cases.push(record("sweeps", n, p, repetitions, || {
            let mut sampler = workload.sampler(1);
            for _ in 0..BINDING_SWEEPS {
                sampler.step();
            }
            black_box(sampler);
        }));

        cases.push(record("fit_progress", n, p, repetitions, || {
            let mut seen = 0usize;
            let fitted = fit_with_progress(&workload.config, &workload.x, &y, 1, |completed, _| {
                seen = completed
            })
            .unwrap();
            black_box((fitted, seen));
        }));
    }

    let doc = serde_json::json!({
        "core_version": thiessen::VERSION,
        "repetitions": repetitions,
        "sweeps": BINDING_SWEEPS,
        "predict_rows": PREDICT_ROWS,
        "cases": cases,
    });
    println!("{}", serde_json::to_string_pretty(&doc).unwrap());
}

/// The predict matrix: the registry's generator at a different seed, so it
/// is not the training design.
fn predict_design(p: usize) -> Data {
    friedman(PREDICT_ROWS, p, SEED.wrapping_add(1)).0
}

fn record(
    case: &str,
    n: usize,
    p: usize,
    repetitions: usize,
    mut run: impl FnMut(),
) -> serde_json::Value {
    let mut seconds: Vec<f64> = (0..repetitions)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64()
        })
        .collect();
    seconds.sort_by(f64::total_cmp);
    serde_json::json!({
        "case": case,
        "n": n,
        "p": p,
        "seconds": seconds[seconds.len() / 2],
        "seconds_min": seconds[0],
        "seconds_max": seconds[seconds.len() - 1],
    })
}
