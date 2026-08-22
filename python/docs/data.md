# Input data

`X` is a two-dimensional numeric array with at least one column and at least
two rows; `y` is numeric with one value per row. The core rejects the following
rather than repairing them, and the message names what is wrong:

| Condition | Error |
| --- | --- |
| a missing or non-finite value in `X` or `y` | `ThiessenError` |
| a constant response | `ThiessenError` |
| a constant column | `ThiessenError` |
| no columns, or fewer than two rows | `ThiessenError` |
| a row count that differs between `X` and `y` | `ThiessenError` |
| a non-integer value in a categorical column | `ThiessenError` |

`ThiessenError` subclasses `ValueError`, so the usual scikit-learn handling
applies.

Imputing missing values is the caller's job. Duplicate rows are valid data. A
response lying exactly on a least-squares fit of the design is valid: the
sigma^2 prior then calibrates from the response standard deviation. More
columns than rows fits and warns.

```python exec="on" source="above" result="text"
import warnings

import numpy as np
from thiessen import Model, TermParams, ThiessenError

rng = np.random.default_rng(0)

try:
    Model(mean_params=TermParams(tessellations=10), burn_in=20, draws=30).fit(
        rng.uniform(size=(20, 2)), np.ones(20)
    )
except ThiessenError as error:
    print("constant response:", error)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Model(mean_params=TermParams(tessellations=10), burn_in=20, draws=30).fit(
        rng.uniform(size=(4, 6)), rng.uniform(size=4), random_state=1
    )
    print("warning:", caught[0].message)
```

At predict the column count must match the fit. An empty matrix is valid.

## Scaling

Euclidean columns are min-max scaled over their training range. Spherical
columns are coordinates in radians and categorical columns are integer level
codes; neither is scaled. A value outside the training range at predict is not
an error: the scaling is fixed at fit and extrapolates.

## Categorical covariates

A categorical covariate reaches the core either as d - 1 indicator columns
under the Euclidean metric, the encoding of `model.matrix` treatment contrasts
and of upstream AddiVortes, or as one column of integer level codes under the
categorical metric. The scikit-learn estimators will do either for you; see
[scikit-learn](sklearn.md).

This section corresponds to rOpenSci general standard G2 and Bayesian standard
BS2.
