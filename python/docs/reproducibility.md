# Reproducibility

The contract is the core crate's, and the crate documentation is its canonical
statement. In short: the same seed, package version and target triple give
identical draws, and the draws never depend on thread count.

## Seeds

`random_state` takes an integer, a `numpy.random.Generator`, a
`numpy.random.RandomState` or `None`, resolved by one rule:

| Given | Resolved to |
| --- | --- |
| an integer | itself, unchanged |
| a `Generator` | one draw from it |
| a `RandomState` | two 32-bit draws from it, combined |
| `None` | fresh entropy from `numpy.random.SeedSequence` |

An integer passes through unchanged, so the same integer reproduces the same
draws through `Model` and through the estimators alike. The resolved seed is on
`FittedModel.random_state` and on the estimators' `random_state_`, so a fit
seeded from `None` can still be repeated.

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(60, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=60)
sweep = {"mean_params": TermParams(tessellations=10), "burn_in": 20, "draws": 40}

first = Model(**sweep).fit(x, y, random_state=None)
repeat = Model(**sweep).fit(x, y, random_state=first.random_state)
print(
    "repeatable from the resolved seed:",
    bool(np.array_equal(first.predict(x), repeat.predict(x))),
)
```

## Across versions and targets

Patch releases of the core crate preserve sampled values for a fixed seed.
Minor releases may change them, and the changelog entry then says "Sampled
values changed" with the reason. `thiessen.CORE_VERSION` reports the core
version an installed wheel was built from, which is the version the contract
attaches to.

The reference target is `x86_64-unknown-linux-gnu`, where fixed-seed chains are
checked against stored snapshots bit for bit. On other targets results are
statistically equivalent and are compared by posterior summaries, never draw by
draw. No claim of bit-exactness is made across targets or across languages.

Transcendental functions go through `libm` rather than the system library, so
the reference target does not drift with libc releases.

## Saving a fit

`FittedModel.save` and `load` carry the core's representation, so a reloaded
model reproduces its predictions exactly rather than approximately. Pickling
the estimators does the same. A file written by one minor version is not
guaranteed to load into another.
