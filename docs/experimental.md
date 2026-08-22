# Experimental items

The stable surface is the method as published: the models and components
of Stone and Gosling (2025) and of CRAN AddiVortes. Everything else the
crate adds is compiled only with the Cargo feature `experimental`, is
tested to the same standard, and is outside the semver promise: its
configuration surface and sampled values may change in any release, with
a changelog line. Enabling the feature does not change the draws of a
configuration that uses no experimental option. The Python and R packages
build the core without the feature.

An item is stabilised when it meets the acceptance criteria of the
contributing guide, has shipped behind the feature for one minor release,
has a stabilisation report on its tracking issue, and has a page under
`docs/` stating the model, priors, and the calibration and recovery
evidence. Removing the gate is a minor version bump.

## Table

| Item | Kind | Configuration | Feature since | Calibration | Status | Tracking issue |
|---|---|---|---|---|---|---|
| Minkowski distance (Manhattan as p = 1) | distance | `geometry.metric` entries `{"minkowski": {"p": ...}}`, `{"manhattan": {}}` | 0.3.0 | conformance, small SBC at p = 1 | experimental | [#61](https://github.com/l-thomson/thiessen/pull/61) |
| Cosine distance | distance | `geometry.metric` entry `{"cosine": {}}` | 0.3.0 | conformance (no triangle inequality), small SBC | experimental | [#62](https://github.com/l-thomson/thiessen/pull/62) |
| Gower distance | distance | `geometry.metric` entries `{"gower": {"kind": "numeric" or "categorical"}}` | 0.3.0 | conformance, small SBC | experimental | [#63](https://github.com/l-thomson/thiessen/pull/63) |
| Mahalanobis distance | distance | `geometry.metric` entry `"mahalanobis"` with `geometry.precision` | 0.3.0 | conformance, small SBC | experimental | [#64](https://github.com/l-thomson/thiessen/pull/64) |
| Per-column composite | distance | `group` label on the minkowski, manhattan, cosine and gower entries | 0.3.0 | conformance per member metric, small SBC | experimental | [#65](https://github.com/l-thomson/thiessen/pull/65) |

Columns: the configuration field or `Model` variant; the first core version
carrying the item behind the feature; calibration status (SBC and Geweke
at the two sizes, or the component conformance tests); experimental or
stabilised, with the core version of stabilisation; the issue carrying the
acceptance checklist and the stabilisation report.

## References

- Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
  Voronoi tessellations. Journal of Computational and Graphical
  Statistics 34(3), 859-871.
