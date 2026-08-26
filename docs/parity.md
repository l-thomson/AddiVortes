# Binding parity

The core's configuration format against its place on each binding's
surface. Rendered by `tools/parity_table.py` from the core's serialised
defaults. The Python suite renders it again and fails on any difference,
and each binding's parity test proves every listed option constructible,
so this file is regenerated, never edited.

The rows are the published surface: an extension built with the core's
`experimental` feature carries the further outcome families of
`docs/experimental.md`, which this table does not list.

Every name is identical across the three surfaces but for the outcome
constructors, which carry an `_outcome` suffix in R: `gaussian()` there
would mask the exported `stats::gaussian()`, and every family takes the
suffix rather than only the one that clashes. The serialised name is
unchanged, so the stored configuration is the same on every surface. Python
groups run-length settings flat on the estimator and `Model`; R groups them
in `general_params()`. The one R shortcut,
`thiessen_control(tessellations =)`, sets `mean_params.tessellations`.

| Core option | Python | R |
| --- | --- | --- |
| `outcome.gaussian.nu` | `gaussian(nu=)` in `outcome=` | `gaussian_outcome(nu = )` in `outcome = ` |
| `outcome.gaussian.q` | `gaussian(q=)` in `outcome=` | `gaussian_outcome(q = )` in `outcome = ` |
| `outcome.probit.offset` | `probit(offset=)` in `outcome=` | `probit_outcome(offset = )` in `outcome = ` |
| `mean_params.tessellations` | `TermParams(tessellations=)` in `mean_params=` | `term_params(tessellations = )` in `mean_params = ` |
| `mean_params.k` | `TermParams(k=)` in `mean_params=` | `term_params(k = )` in `mean_params = ` |
| `mean_params.lambda_c` | `TermParams(lambda_c=)` in `mean_params=` | `term_params(lambda_c = )` in `mean_params = ` |
| `mean_params.geometry.metric` | `GeometryParams(metric=)` in `mean_params=` | `geometry_params(metric = )` in `mean_params = ` |
| `mean_params.geometry.sigma_c` | `GeometryParams(sigma_c=)` in `mean_params=` | `geometry_params(sigma_c = )` in `mean_params = ` |
| `mean_params.structure.omega` | `StructureParams(omega=)` in `mean_params=` | `structure_params(omega = )` in `mean_params = ` |
| `variance_params.tessellations` | `TermParams(tessellations=)` in `variance_params=` | `term_params(tessellations = )` in `variance_params = ` |
| `variance_params.k` | `TermParams(k=)` in `variance_params=` | `term_params(k = )` in `variance_params = ` |
| `variance_params.lambda_c` | `TermParams(lambda_c=)` in `variance_params=` | `term_params(lambda_c = )` in `variance_params = ` |
| `variance_params.geometry.metric` | `GeometryParams(metric=)` in `variance_params=` | `geometry_params(metric = )` in `variance_params = ` |
| `variance_params.geometry.sigma_c` | `GeometryParams(sigma_c=)` in `variance_params=` | `geometry_params(sigma_c = )` in `variance_params = ` |
| `variance_params.structure.omega` | `StructureParams(omega=)` in `variance_params=` | `structure_params(omega = )` in `variance_params = ` |
| `general_params.burn_in` | `burn_in=` on `Model` and the estimators | `general_params(burn_in = )` |
| `general_params.draws` | `draws=` on `Model` and the estimators | `general_params(draws = )` |
| `general_params.thinning` | `thinning=` on `Model` and the estimators | `general_params(thinning = )` |
| `general_params.prior_only` | `prior_only=` on `Model` and the estimators | `general_params(prior_only = )` |

Groups without a row: `mean_params.cell` and `variance_params.cell` carry
no field on the stable surface; the within-cell basis is experimental and
core-only (`docs/experimental.md`).

The seed is not part of the configuration: it is `random_state` in Python
and `seed` in R, resolved by each language's rule and passed to the core
unchanged.
