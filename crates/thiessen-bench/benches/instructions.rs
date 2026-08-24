//! Instruction-count benchmarks under callgrind: deterministic counts for
//! the one measurement a shared runner can make honestly.
//!
//! No other measurement in the repository is read by a gate. Wall-clock on
//! a shared runner sits at a few per cent of noise with far larger
//! excursions, and code layout alone moves it by around eight per cent
//! (Mytkowicz, Diwan, Hauswirth and Sweeney 2009); instruction counts move
//! only when the work does.
//!
//! Requires valgrind and the runner:
//!
//! ```text
//! cargo install --version 0.19.4 gungraun-runner
//! cargo bench -p thiessen-bench --bench instructions
//! ```
//!
//! The attribute macros take literal case names, so this file does not
//! iterate the registry the way the criterion benchmarks do. The
//! assertions at the foot fail the build when a model or a column count is
//! added to the registry and not listed here.

use std::hint::black_box;

use gungraun::{
    library_benchmark, library_benchmark_group, main, Callgrind, EventKind, LibraryBenchmarkConfig,
};
use thiessen::{Data, Fitted, Sampler};
use thiessen_bench::{Case, CASES, N, P, SCALING_P, SEED};

/// The registry entry of `name`.
///
/// # Panics
///
/// If the registry carries no such model, which is a rename that has not
/// reached this file.
fn case(name: &str) -> &'static Case {
    CASES
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no registry case named {name}"))
}

fn sampler(name: &str) -> Sampler {
    (case(name).build)(N, P).sampler(SEED)
}

fn scaling_sampler(p: usize) -> Sampler {
    thiessen_bench::scaling(p).sampler(SEED)
}

fn fitted(name: &str) -> (Fitted, Data) {
    let workload = (case(name).build)(N, P);
    let fitted = workload.fit(SEED);
    (fitted, workload.x)
}

#[library_benchmark]
#[benches::models(args = ["gaussian", "probit", "heteroscedastic"], setup = sampler)]
fn sweep(mut sampler: Sampler) {
    sampler.step();
    black_box(sampler);
}

// A tessellation reads a handful of columns, so the count is near-flat in
// p. The growth an all-column distance loop produces shows up here as a
// rising instruction count while the wall-clock benchmarks still look
// like noise.
#[library_benchmark]
#[benches::columns(args = [5, 10, 40], setup = scaling_sampler)]
fn sweep_scaling(mut sampler: Sampler) {
    sampler.step();
    black_box(sampler);
}

#[library_benchmark]
#[benches::models(args = ["gaussian", "probit", "heteroscedastic"], setup = fitted)]
fn predict(input: (Fitted, Data)) {
    let (fitted, x) = input;
    black_box(fitted.predict(&x).unwrap());
}

library_benchmark_group!(
    name = counts;
    benchmarks = sweep, sweep_scaling, predict
);

main!(
    // Five per cent on retired instructions and on estimated cycles. Below
    // that the counts still move with allocator and libc differences
    // between toolchains; above it, the work itself has changed.
    config = LibraryBenchmarkConfig::default().tool(
        Callgrind::default().soft_limits([
            (EventKind::Ir, 5f64),
            (EventKind::EstimatedCycles, 5f64),
        ])
    );
    library_benchmark_groups = counts
);

// The macro arguments above are literals; these fail the build when the
// registry grows and this file has not kept up.
const _: () = assert!(cfg!(feature = "experimental") || CASES.len() == 3);
const _: () = assert!(SCALING_P.len() == 3);
