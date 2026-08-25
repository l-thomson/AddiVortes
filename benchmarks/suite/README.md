# Benchmark suite

A fixed set of cells covering every shipped model, and the scorecard of
each. The cells come from the core's benchmark registry
(`bench/src/lib.rs`) through `suite list`, so a model added there is
benchmarked without an edit here.

## Running it

    pip install -r benchmarks/suite/requirements.txt
    python benchmarks/suite/run.py --sizes small

The driver builds the release binary, runs each chain in its own process,
and writes a tidy CSV: one row per cell and metric, with a value, a
standard error over repetitions, and the repetition count behind it. It
reads the peak resident set from the child process through `os.wait4`, so
it needs a POSIX host; the numbers are comparable only within one machine
in any case.

Repetitions are sized from the variance observed so far. The driver keeps
adding repetitions of a cell until the standard error of its minimum bulk
ESS per second falls below `--target-rse`, and stops at `--reps-max`
regardless.

Comparing two scorecards:

    python benchmarks/suite/compare.py baselines/core-v0.3.0.csv new.csv

Every metric comes back as a ratio with a 95 per cent confidence interval
built on the log ratio from the two standard errors (Kalibera and Jones
2013). The `separated` column says whether the interval excludes one. A
point ratio without an interval is not a result.

## The schedule

Cells run 500 burn-in sweeps and keep 4000 draws per chain, four chains
per repetition. That is longer than the shipped default of 1000 draws,
which leaves rank-normalised R-hat around 1.035 on held-out f(x) for the
Gaussian model at these sizes. Efficiency measured on a chain that has
not converged describes nothing, so the suite pays for convergence and
reports the cost as part of the measurement.

## What is measured

The currency is minimum effective sample size per second over the
inferential quantities, bulk and tail, with R-hat as a validity gate.
Wall-clock alone is the wrong currency for a Markov chain: a sampler
twice as fast per sweep that mixes half as well has gained nothing. ESS
per sweep is reported beside ESS per second, so algorithmic efficiency
stays separable from implementation speed and the pair survives a change
of hardware; the seconds are post-warm-up with initialisation excluded,
which is the pair bartz reports (arXiv:2410.23244, appendix C.1).

Declared quantities are sigma where the model samples one, f(x) at five
held-out rows, and the two structure counts. The structure counts are
reported and take no part in the currency or the validity gate: a sum of
tessellations is not identified, so the structure of a draw wanders while
the function it encodes does not.

Every diagnostic comes from one pinned ArviZ: rank-normalised split-chain
R-hat, bulk ESS and tail ESS (Vehtari, Gelman, Simpson, Carpenter and
Bürkner 2021). Nothing in this directory or in the core estimates an
effective sample size for a benchmark. ESS estimators disagree materially
on poorly mixed chains, so which one produced a number is part of the
number.

Beside those: held-out RMSE, log predictive density, the coverage and
mean width of the 95 per cent predictive interval, seconds to a bulk ESS
of 400, wall-clock fit and predict, peak resident set, and the largest
Monte Carlo standard error over the inferential quantities.

## Baselines

`baselines/` holds one committed scorecard per core release, named for
the version it was taken at. Comparisons are recomputed on demand from
those files. There is no dashboard and no service: the pattern with
documented rot is the scheduled dashboard, and a file in the repository
cannot go offline.

A baseline is refreshed at a release, and before a snapshot regeneration,
by dispatching the CI workflow with the sizes and repetitions wanted and
committing the artefact it uploads.

## What the gate reads

The `suite` job compares against the newest committed baseline and fails
on a cell that stops converging, or on a separated adverse move in
effective sample size per sweep or in held-out error. Convergence is read
against the baseline rather than against the limit outright: a cell whose
R-hat already sat above the limit is a property of the sampler at that
schedule, not something a pull request did.

Effective sample size per sweep and held-out error are ratios per sweep
and per row: they do not change with the speed of the machine, so a
shared runner can be asked about them. No wall-clock metric is gated,
here or anywhere else in the repository.
