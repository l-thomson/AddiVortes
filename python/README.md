# thiessen (Python)

Python bindings to the `thiessen` Rust crate, an implementation of
AddiVortes: Bayesian regression on a sum of Voronoi tessellations (Stone and
Gosling, 2025, *Journal of Computational and Graphical Statistics* 34(3),
859-871, [doi:10.1080/10618600.2024.2414104](https://doi.org/10.1080/10618600.2024.2414104)).

AddiVortes replaces the decision trees of BART (Chipman, George and
McCulloch, 2010) with Voronoi tessellations, so a cell is a region of the
covariate space rather than a box. The sampler is the Gibbs sampler of the
paper; the models are Gaussian regression, binary probit and the
heteroscedastic variant.

## Installation

The package builds the core crate from source, so a Rust toolchain
(`cargo`, `rustc` >= 1.74) is required until wheels are published.

```sh
pip install -e python/
```

## Usage

```python
import numpy as np
from thiessen import Model

rng = np.random.default_rng(0)
x = rng.uniform(size=(200, 3))
y = 3.0 * (x[:, 0] - 0.4) ** 2 + 0.5 * x[:, 1] + rng.normal(scale=0.1, size=200)

fitted = Model(m=50, burn_in=100, draws=200).fit(x, y, random_state=1)
mean = fitted.predict(x)
lower, upper = fitted.credible_interval(x, level=0.9).T
```

`random_state` takes an integer, a `numpy.random.Generator`, a
`numpy.random.RandomState` or `None`. The same integer, package version and
target reproduce the same draws; the resolved seed is on
`fitted.random_state`.

## scikit-learn

With the `sklearn` extra, `thiessen.estimators` holds two estimators meeting
the scikit-learn contract, so they compose with `Pipeline`, `GridSearchCV`,
`cross_val_score` and `sklearn.inspection`.

```python
from sklearn.inspection import PartialDependenceDisplay
from thiessen.estimators import AddiVortesRegressor

model = AddiVortesRegressor(m=50, burn_in=100, draws=200, random_state=1)
model.fit(x, y)
PartialDependenceDisplay.from_estimator(model, x, features=[0, 1])
```

`AddiVortesClassifier` fits the binary probit model and carries
`predict_proba`. An integer `random_state` gives the same draws through either
API, so `AddiVortesRegressor(random_state=1)` and `Model(random_state=1)`
agree.

### Priors

| Parameter | Default | Stone and Gosling (2025) |
| --- | --- | --- |
| `m` | 200 | 200 |
| `nu` | 6 | 6 |
| `q` | 0.85 | 0.85 |
| `k` | 3 | 3 |
| `sigma_c` | 0.8 | 0.8 |
| `omega` | min(3, p) | min(3, p) |
| `lambda_c` | 5 | 25 |

The `lambda_c` default follows AddiVortes >= 0.6.8; pass `lambda_c=25` for the
paper's value.

### Categorical covariates

`categorical_features` follows the `HistGradientBoostingRegressor` shape.
Without it the input is taken as numeric, and the usual route is an explicit
encoder:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ColumnTransformer([("g", OneHotEncoder(drop="first"), ["group"])])
```

Naming the columns instead has the estimator apply the same d - 1
treatment-contrast encoding, the first level as reference, as upstream
AddiVortes and `model.matrix`. A column whose `metric` entry is
`'categorical'` passes as integer level codes and takes the Eskin mismatch
weight rather than being expanded.

## arviz

With the `arviz` extra, a fit converts to an arviz `DataTree`:

```python
import arviz as az

data = fitted.to_inference_data(x, y)
az.summary(data, var_names=["sigma"])
az.loo(data)
```

The sampler runs one chain, so every group has a chain dimension of one.

## Models and the stable surface

`model` takes `gaussian`, `probit` or `heteroscedastic`, the models of Stone
and Gosling (2025) and of CRAN AddiVortes. Everything else the core crate adds
sits behind its `experimental` Cargo feature, which this package does not
enable, so a configuration or a saved model naming such an option is rejected
with the core's message naming the feature. The table of experimental items and
their status is `docs/experimental.md` in the repository. A graduated item is
exposed here as any other option, with no separate opt-in.

## Documentation

The reproducibility contract, the input-data contract and the testing
strategy are in the repository: `README.md`, `docs/models.md` and
`docs/testing.md`.

## Licence

MIT or Apache-2.0, at your option.
