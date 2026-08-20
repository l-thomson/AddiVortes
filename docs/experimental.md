# Experimental items

The stable surface is the method as published: the models and components
of Stone and Gosling (2025) and of CRAN AddiVortes. Everything else the
crate adds is compiled only with the Cargo feature `experimental`, is
tested to the same standard, and is outside the semver promise: its
configuration surface and sampled values may change in any release, with
a changelog line. Enabling the feature does not change the draws of a
configuration that uses no experimental option. The Python and R packages
build the core without the feature.

An item graduates to the stable surface when it meets the model or
component bar of the contributing guide, has a citable write-up with a
DOI stating the model, priors, calibration and recovery evidence, and has
shipped behind the feature for one minor release with its tracking issue
closed. Removing the gate is a minor version bump.

## Table

| Item | Kind | Configuration | Feature since | Calibration | Status |
|---|---|---|---|---|---|

Columns: the configuration field or `Model` variant; the first core version
carrying the item behind the feature; calibration status (SBC and Geweke
at the two sizes, or the component conformance tests); experimental or
graduated, with the core version of graduation.

## References

- Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
  Voronoi tessellations. Journal of Computational and Graphical
  Statistics 34(3), 859-871.
