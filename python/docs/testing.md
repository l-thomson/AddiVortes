# Testing

The testing strategy is shared with the core crate and the R package, and its
canonical statement is `docs/testing.md` in the
[repository](https://github.com/l-thomson/thiessen/blob/dev/docs/testing.md).
It runs from unit tests through known-answer tests to simulation-based
calibration and Geweke tests, with comparison fixtures against CRAN AddiVortes
and fixed-seed snapshots on the reference target.

What this package adds on top:

| Test | What it holds |
| --- | --- |
| seed parity | the package's draws equal the core's stored Gaussian chain, bit for bit on the reference target |
| API parity | the estimators' parameter names are a subset of the core's configuration fields |
| seed rule | an integer reaches the core unchanged, so `Model` and the estimators draw alike |
| stub sync | `_native.pyi` declares every name the extension exposes, and no others |
| estimator contract | `sklearn.utils.estimator_checks.parametrize_with_checks`, with no expected failures |
| exposure policy | every family the core carries has a constructor, and a family the extension gates raises `RequiresFeatureError` |

The seed parity test parses the core's snapshot file rather than holding a copy
of it, so a regenerated snapshot cannot leave the two out of step.

Reproducing the core's fixture in numpy needs the multiplications to associate
as the core's do: `3.0 * d ** 2` rounds differently from `(3.0 * d) * d`, and
one flipped bit in the response moves the whole chain.
