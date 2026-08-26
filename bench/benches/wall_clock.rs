//! Wall-clock benchmarks: one sampler sweep and one predict call per
//! shipped model, the fit's growth in p, and the threaded chains and
//! prediction against their one-thread cost.
//!
//! Read through `tools/perf-compare.sh`, which runs this against two
//! revisions on one machine in one session and reports the paired deltas.
//! Wall-clock numbers from separate machines or separate sessions are not
//! comparable and no gate reads them.
//!
//! The sweep benchmarks step one persistent sampler rather than building a
//! fresh one per iteration: construction costs more than a sweep and would
//! swamp it, and the state after criterion's warm-up is the stationary
//! regime the measurement is about.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use thiessen::{fit_chains_with_threads, IntervalKind};
use thiessen_bench::{Case, Workload, CASES, N, P, SCALING_P, SEED};

/// Threads of the threaded benchmarks; the chain count too, so every
/// chain has a thread of its own.
const THREADS: usize = 4;

/// The registry entry of `name`.
///
/// # Panics
///
/// If the registry carries no such model.
fn case(name: &str) -> &'static Case {
    CASES
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no registry case named {name}"))
}

fn sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("sweep");
    for case in CASES {
        let workload = (case.build)(N, P);
        let mut sampler = workload.sampler(SEED);
        group.bench_function(case.name, |b| b.iter(|| sampler.step()));
    }
    let spherical = thiessen_bench::spherical(N);
    let mut sampler = spherical.sampler(SEED);
    group.bench_function("spherical", |b| b.iter(|| sampler.step()));
    group.finish();
}

fn predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict");
    for case in CASES {
        let workload = (case.build)(N, P);
        let fitted = workload.fit(SEED);
        group.bench_function(case.name, |b| {
            b.iter(|| black_box(fitted.predict(&workload.x).unwrap()))
        });
    }
    group.finish();
}

/// The registry's short schedule as `THREADS` chains on one thread and on
/// `THREADS` threads, and the Gaussian predict over the same thread
/// counts. The one-thread rows are the serial cost the threaded rows are
/// read against; both are wall-clock only, so no gate reads them.
fn threads(c: &mut Criterion) {
    let mut group = c.benchmark_group("threads");
    group.sample_size(10);
    let workload: Workload = (case("gaussian").build)(N, P);
    for n_threads in [1, THREADS] {
        group.bench_with_input(
            BenchmarkId::new("chains", n_threads),
            &n_threads,
            |b, &n_threads| {
                b.iter(|| {
                    black_box(
                        fit_chains_with_threads(
                            &workload.config,
                            &workload.x,
                            workload.numeric(),
                            SEED,
                            THREADS,
                            n_threads,
                        )
                        .unwrap(),
                    )
                })
            },
        );
    }
    let mut fitted = workload.fit(SEED);
    let wide = (case("gaussian").build)(20 * N, P);
    for n_threads in [1, THREADS] {
        fitted.set_threads(n_threads);
        group.bench_with_input(
            BenchmarkId::new("predict", n_threads),
            &n_threads,
            |b, _| b.iter(|| black_box(fitted.predict(&wide.x).unwrap())),
        );
    }
    group.finish();
}

/// The posterior mean and a credible interval over the same draws, as two
/// calls and as the one-traversal call. The pair is the cost of the second
/// traversal, which is the shape of the gap against upstream's single
/// compiled traversal.
fn predict_interval(c: &mut Criterion) {
    let mut group = c.benchmark_group("predict_interval");
    let workload: Workload = (case("gaussian").build)(N, P);
    let fitted = workload.fit(SEED);
    group.bench_function("mean_then_interval", |b| {
        b.iter(|| {
            black_box(fitted.predict(&workload.x).unwrap());
            black_box(fitted.credible_interval(&workload.x, 0.95).unwrap());
        })
    });
    group.bench_function("with_interval", |b| {
        b.iter(|| {
            black_box(
                fitted
                    .predict_with_interval(&workload.x, IntervalKind::Credible, 0.95)
                    .unwrap(),
            )
        })
    });
    group.finish();
}

/// One sweep of the Gaussian model at each column count. A tessellation
/// reads a handful of columns, so these three should sit within a few per
/// cent of each other; growth in p is a defect in the distance path.
fn scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_p");
    for p in SCALING_P {
        let workload = thiessen_bench::scaling(p);
        let mut sampler = workload.sampler(SEED);
        group.bench_with_input(BenchmarkId::from_parameter(p), &p, |b, _| {
            b.iter(|| sampler.step())
        });
    }
    group.finish();
}

/// The per-sweep progress callback, against the same fit without one. The
/// bindings call across the language boundary here once per sweep, so the
/// cost of an empty callback is the floor that binding measurement is read
/// against.
fn progress(c: &mut Criterion) {
    let mut group = c.benchmark_group("progress");
    let workload = (case("gaussian").build)(N, P);
    let y = workload.numeric();
    group.sample_size(20);
    group.bench_function("off", |b| {
        b.iter(|| black_box(thiessen::fit(&workload.config, &workload.x, y, 1).unwrap()))
    });
    group.bench_function("on", |b| {
        b.iter(|| {
            let mut seen = 0usize;
            let fitted =
                thiessen::fit_with_progress(&workload.config, &workload.x, y, 1, |completed, _| {
                    seen = completed
                })
                .unwrap();
            black_box((fitted, seen))
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    sweep,
    predict,
    predict_interval,
    scaling,
    progress,
    threads
);
criterion_main!(benches);
