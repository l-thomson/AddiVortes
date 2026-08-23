# Experimental items

The stable surface is the method as published: the models and components
of Stone and Gosling (2025) and of CRAN AddiVortes. Everything else the
crate adds is compiled only with the Cargo feature `experimental`, is
tested to the same standard, and is outside the semver promise: its
configuration surface and sampled values may change in any release, with
a changelog line. Enabling the feature does not change the draws of a
configuration that uses no experimental option. The Python and R packages
build the core without the feature.

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

Columns: the configuration field or variant; the first core version
carrying the item behind the feature; calibration status (SBC and Geweke
at the two sizes, or the component conformance tests); experimental or
stabilised, with the core version of stabilisation; the pull request that
added the item, its public record.

## References

- Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
  Voronoi tessellations. Journal of Computational and Graphical
  Statistics 34(3), 859-871.
