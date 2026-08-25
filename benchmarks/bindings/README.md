# Binding performance

Four cases, in both bindings, against the core's time on the same work.
Nothing here changes sampled values, nothing is stored, and nothing gates.
Run it on one machine, old revision against new, and put the table in the
pull request: the pull request is the record and its history is the
archive.

## The four cases

- One `fit` call. Amortised over the whole schedule, so a slope across
  sizes here means the design or the fitted state is being copied.
- One `predict` call on a large matrix: return-value marshalling.
- N sampler `step` calls, per call against batched. The sampler API is
  callable from both bindings, so a user's loop crosses the boundary once
  per sweep instead of once per fit. `step(n)` runs n sweeps behind one
  crossing and `step()` called n times runs them behind n crossings, so
  the pair measures the boundary cost with the sampling held identical.
- `fit` with progress reporting on against off: the per-sweep callback. R
  only, since the Python binding has no progress surface and nothing
  crosses back once per sweep there.

Each over three sizes. A copy of the design appears as a slope and not as
an offset, so one baseline, one that grows in rows and one that grows in
columns are needed to tell them apart.

## Running it

The designs and the core's own times come from the core, so both sides run
on the same numbers rather than on two generators that happen to share a
name:

    cargo run --release --manifest-path bench/Cargo.toml --bin overhead -- \
        designs target/bindings
    cargo run --release --manifest-path bench/Cargo.toml --bin overhead -- \
        run > target/bindings/core.json

R, with `bench`, which tracks time, allocations and garbage collections
and verifies equality of the compared expressions by default. It needs
`bench`, `jsonlite` and `progressr` beside the installed package, none of
which the package itself depends on:

    Rscript -e 'install.packages(c("bench", "jsonlite", "progressr"))'
    Rscript benchmarks/bindings/overhead.R target/bindings

Python, with pytest-benchmark:

    pip install -r python/requirements-bench.txt
    cd python
    pytest tests/benchmarks --benchmark-json=../target/bindings/python.json
    cd ..
    python benchmarks/bindings/overhead.py target/bindings

Both tables report elapsed time, the core's time on the same case, the
absolute difference and the ratio. The absolute difference is there
because a ratio on a call of a few hundred microseconds carries no
information. `profvis` in R and `memray` in Python are the tools for
locating a cost once a number looks wrong; neither is part of the
measurement.

## What is absent, and why

No wall-clock in CI, no gate, no committed result files, no hosted
service, and no single mechanism spanning both languages. Runner variance
is the reason: relative measurement inside one job exists precisely
because shared runners vary by up to roughly thirty per cent, and the
epic already rules out wall-clock gates on them.

Evaluated and rejected:

- `benchmark-action/github-action-benchmark` with `external-data-json-path`.
  Committed results per pull request and an alert comment, but the R path
  needs a hand-written converter from `bench::mark` to its custom JSON
  schema, and what gets committed is shared-runner wall-clock: the file
  reads as authority while carrying none.
- touchstone. The right shape for R, and both branches in one job with
  randomised order and significance reported, but not on CRAN, one
  maintainer, and it installs both branches, so the vendored core compiles
  twice per run.
- CodSpeed with pytest-codspeed. Trusted, used by Polars and pydantic, and
  its instruction counting works through the interpreter, which the
  callgrind path in `bench/` does not. Hosted third party and Python only.
  Worth reconsidering for Python alone if the reporting habit proves
  insufficient.
- asv and Conbench. The dashboard and service patterns the epic rejects
  for documented rot; Arrow's own Conbench runs have broken on installing
  an R benchmark dependency.
