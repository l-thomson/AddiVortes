# Comparator scripts and release evidence

Reproducible comparisons of this library against upstream AddiVortes,
dbarts, BART, stochtree and an XGBoost accuracy baseline. Run by hand at
releases, after the batched fixture regeneration, so the published
numbers describe the released code. Nothing here runs in CI.

This directory is the reproduction package for the paper's performance
claims: every published table and figure is a script over one CSV, and a
reviewer who runs the commands below on their own machine rebuilds the
claims from their own measurements.

## Environments

Both environments are pinned; there is no container.

Python, through [uv](https://docs.astral.sh/uv/):

```sh
uv venv .venv && VIRTUAL_ENV=.venv uv pip install -r benchmarks/comparators/requirements.txt
VIRTUAL_ENV=.venv uv pip install ./python   # this library, from the release tag
```

`requirements.txt` is compiled from `requirements.in` by
`uv pip compile`; regenerate it the same way when a pin moves.

R, through [renv](https://rstudio.github.io/renv/): `renv.lock` pins
AddiVortes, dbarts, BART and jsonlite with their dependencies. Restore
into a library and put that library on the search path:

```sh
Rscript -e 'renv::restore(lockfile = "benchmarks/comparators/renv.lock", library = "rlib")'
export R_LIBS_USER="$PWD/rlib:$R_LIBS_USER"
```

## Running

```sh
python benchmarks/comparators/run.py --out target/comparators
python benchmarks/comparators/run.py --grid scaling --out target/comparators-scaling
python benchmarks/comparators/parity.py target/comparators/comparison.csv
python benchmarks/comparators/analyse.py target/comparators/comparison.csv
python benchmarks/comparators/analyse.py target/comparators-scaling/comparison.csv
```

The main grid is six methods over three generated processes at three
sizes, plus two real datasets, at three seeds and more where the observed
variance asks (`run.py` adds seeds to a cell group until the relative
standard error of its fit time is below 5 per cent, capped at six). The
scaling grid sweeps n, p and the ensemble size m one at a time on
Friedman #1, everything else held at the base point.

Each cell runs in its own subprocess, in a shuffled order under a fixed
seed, after one discarded warm-up cell, with every thread-count
environment variable set to one. A cell whose user time exceeds its
elapsed time, or whose held-out error is more than 1.5 times the best on
the same data, is flagged in the CSV. The parity gate must pass before
any timing against upstream is quoted: realised cells per tessellation
within 15 per cent and held-out summaries within four standard errors,
because two implementations that fit different models cannot be timed
against each other. Upstream's cell-count prior thins by 1 / (b + 1), so
agreement on the nominal parameter is not agreement on the model.

The output is one tidy CSV (method by dataset by seed by metric) plus a
`meta.json` per cell with versions, seeds, platform, peak resident set
and the exact schedule. Tables and figures are scripts over the CSV;
none of them measures anything.

## Metric policy

The diagnostics come from the one pinned ArviZ in `requirements.txt`,
computed over every method's draws alike; a package's own diagnostics
are never quoted. Cross-method efficiency is computed on common
quantities only, f(x) at held-out rows and the log predictive density,
and is reported beside RMSE, coverage and interval width so sampling
efficiency is not conflated with model fit. ESS per sweep and post
warm-up seconds per sweep sit beside ESS per second, separating
algorithmic efficiency from implementation speed. Seconds to an ESS of
400, warm-up included, is the practitioner's number. Every published
ratio carries a 95 per cent interval on the log scale with
Welch-Satterthwaite degrees of freedom (Kalibera and Jones 2013), and
scaling claims are fitted exponents with intervals, not point ratios.

Only stable models appear. Nothing behind the `experimental` Cargo
feature reaches the CSV or the tables.

## Evaluated and rejected

- Spatial indexes for the cell-assignment inner loop. At the realised
  cell counts of this model family (single digits per tessellation under
  the thinned prior) a k-d tree or ball tree costs more to build and
  traverse than the linear scan it replaces. Measured before rejection;
  the crossover sits far above the sizes any published cell reaches.
- GPU execution. Below roughly 10^5 rows the transfer and launch
  overhead exceeds the arithmetic; every published cell is under that,
  so a GPU column would compare host-device copies, not methods.
- A single cross-model "ESS/sec on everything" number. Model-specific
  quantities (one residual variance against a variance ensemble, cell
  counts against tree depths) make such a number a comparison of
  parameterisations. Hence the common-quantities rule above, and the fit
  metrics beside the efficiency metrics rather than folded into them.
- Wall-clock gates in CI. Shared runners vary too much for a 5 per cent
  claim; the instruction-count gate in `bench/` covers regressions, and
  this directory covers claims, by hand, on one machine.

## Claims and the SIGPLAN checklist

Each published claim maps to the
[SIGPLAN empirical evaluation checklist](https://www.sigplan.org/Resources/EmpiricalEvaluation/):

- Clearly stated claims: each table states its metric, schedule and
  dataset in the caption; the schedule string is in every CSV row.
- Suitable comparison points: upstream at the pinned CRAN version, the
  two standard BART implementations, the newer stochtree, and a
  non-Bayesian accuracy baseline; all through their released packages.
- Suitable benchmarks: Friedman #1 plus a process with an oblique
  boundary and one with correlated covariates, because Friedman alone
  favours tree models, and two standard real datasets.
- Adequate data analysis: repetitions sized from observed variance,
  standard errors across seeds, ratio intervals per Kalibera and Jones,
  exponents with confidence intervals for scaling.
- Relevant metrics: time to target ESS rather than raw wall clock;
  accuracy beside efficiency.
- Appropriate experimental design: one subprocess per cell, shuffled
  order, discarded warm-up, single-thread pinning with a guard, a
  held-out accuracy guard, the parity gate before upstream timings.
- Presentation of results: every number in the tables is rebuildable
  from `comparison.csv` by `analyse.py`; the CSV and per-cell metadata
  ship with the release.
