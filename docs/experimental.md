# Experimental items

The stable surface is the method as published: the models and components
of Stone and Gosling (2025) and of CRAN AddiVortes. Everything else the
crate adds is compiled only with the Cargo feature `experimental`, is
tested to the same standard, and is outside the semver promise: its
configuration surface and sampled values may change in any release, with
a changelog line. Enabling the feature does not change the draws of a
configuration that uses no experimental option.

The Python and R packages build the core without the feature by default,
and every released artefact ships that way. The opt-in is a build-time
one, since the feature is a `#[cfg]` in the core and an experimental
item's sampled values may change in a patch release:

```sh
THIESSEN_EXPERIMENTAL=1 R CMD INSTALL r
pip install ./python --config-settings build-args="--features experimental"
```

A binding reports its own setting, `core_experimental()` in R and
`thiessen._native.EXPERIMENTAL` in Python. A configuration or a saved fit
naming a gated item in a build without the feature is rejected by
`Config::validate` with `Error::RequiresFeature`, which reaches R as the
condition class `thiessen_requires_feature` and Python as
`RequiresFeatureError`. The outcome families have a constructor in both
bindings whatever the build carries, so a script naming one is portable.
A fit saved from a build with the feature does not load in one without
it.

The stabilisation rule is stated once, in the crate-root documentation
(`crates/thiessen/src/lib.rs`, Stability): graduation is a pull request
against that rule, not a ticket. This file is the status table for every
gated item; the pull-request column is the public record of each item's
history.

## Table

| Item | Kind | Configuration | Feature since | Calibration | Status | Pull request |
|---|---|---|---|---|---|---|
| Minkowski distance (Manhattan as p = 1) | distance | `geometry.metric` entries `{"minkowski": {"p": ...}}`, `{"manhattan": {}}` | 0.3.0 | conformance, small SBC at p = 1 | experimental | [#61](https://github.com/l-thomson/thiessen/pull/61) |
| Cosine distance | distance | `geometry.metric` entry `{"cosine": {}}` | 0.3.0 | conformance (no triangle inequality), small SBC | experimental | [#62](https://github.com/l-thomson/thiessen/pull/62) |
| Gower distance | distance | `geometry.metric` entries `{"gower": {"kind": "numeric" or "categorical"}}` | 0.3.0 | conformance, small SBC | experimental | [#63](https://github.com/l-thomson/thiessen/pull/63) |
| Mahalanobis distance | distance | `geometry.metric` entry `"mahalanobis"` with `geometry.precision` | 0.3.0 | conformance, small SBC | experimental | [#64](https://github.com/l-thomson/thiessen/pull/64) |
| Per-column composite | distance | `group` label on the minkowski, manhattan, cosine and gower entries | 0.3.0 | conformance per member metric, small SBC | experimental | [#65](https://github.com/l-thomson/thiessen/pull/65) |
| Weighted inclusion | inclusion prior | `structure.inclusion` entry `{"weighted": {"weights": [...]}}` | 0.3.0 | conformance, small SBC | experimental | [#66](https://github.com/l-thomson/thiessen/pull/66) |
| DART inclusion | inclusion prior (model-grade validation) | `structure.inclusion` entry `{"dart": {"a": ..., "b": ..., "rho": ...}}` | 0.3.0 | SBC and Geweke, both sizes; broken-sampler fixture | experimental | [#67](https://github.com/l-thomson/thiessen/pull/67) |
| Linear cell basis | cell basis (model-grade validation) | `mean_params.cell.basis` entry `"linear"` | 0.3.0 | known answer; SBC and Geweke, both sizes; broken-sampler fixture | experimental | [#68](https://github.com/l-thomson/thiessen/pull/68) |
| Soft membership | membership rule (model-grade validation) | `mean_params.geometry.membership` entry `{"soft": {"rate": ...}}` | 0.3.0 | known answer; SBC and Geweke, both sizes; broken-sampler fixture | experimental | [#78](https://github.com/l-thomson/thiessen/pull/78) |
| Tobit outcome | outcome model (model-grade validation) | `outcome` entry `{"tobit": {"lower": ..., "upper": ...}}` | 0.3.0 | known answer (censored-likelihood quadrature); SBC and Geweke, both sizes | experimental | [#81](https://github.com/l-thomson/thiessen/pull/81) |
| AFT outcome | outcome model (model-grade validation) | `outcome` entry `{"aft": {}}` with `fit_aft(x, times, events)` | 0.3.0 | known answer (censored-likelihood quadrature); SBC and Geweke, both sizes; informational `abart` comparison | experimental | [#82](https://github.com/l-thomson/thiessen/pull/82) |
| Interval-censored outcome | outcome model (model-grade validation) | `outcome` entry `{"interval_censored": {}}` with `fit_interval_censored(x, lower, upper)` | 0.3.0 | known answer (interval-likelihood quadrature); SBC and Geweke, both sizes | experimental | [#83](https://github.com/l-thomson/thiessen/pull/83) |
| Ordinal outcome | outcome model (model-grade validation) | `outcome` entry `{"ordinal": {"categories": ...}}` | 0.3.0 | known answer (cutpoint and cell-mean quadrature); SBC and Geweke, both sizes, cutpoints covered; broken-sampler fixture; full-size cutpoint ESS check | experimental | [#85](https://github.com/l-thomson/thiessen/pull/85) |
| Student-t outcome | outcome model (model-grade validation) | `outcome` entry `{"student_t": {"df": 4.0}}` or `{"student_t": {"df": [3.0, 6.0, 12.0]}}` | 0.3.0 | known answer (marginal t-likelihood quadrature, fixed and grid df); SBC and Geweke, both sizes (fixed df); no acceptance-ratio term, so no broken-sampler fixture | experimental | [#98](https://github.com/l-thomson/thiessen/pull/98) |
| Laplace outcome | outcome model (model-grade validation) | `outcome` entry `{"laplace": {}}` | 0.3.0 | known answer (marginal Laplace-likelihood quadrature); SBC and Geweke, both sizes; no acceptance-ratio term, so no broken-sampler fixture | experimental | [#99](https://github.com/l-thomson/thiessen/pull/99) |

Columns: the configuration field or variant; the first core version
carrying the item behind the feature; calibration status (SBC and Geweke
at the two sizes, or the component conformance tests); experimental or
stabilised, with the core version of stabilisation; the pull request that
added the item, its public record.

## References

- Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
  Voronoi tessellations. Journal of Computational and Graphical
  Statistics 34(3), 859-871.
