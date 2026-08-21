# thiessen

Python bindings to the `thiessen` Rust crate, an implementation of AddiVortes:
Bayesian regression on a sum of Voronoi tessellations (Stone and Gosling, 2025,
*Journal of Computational and Graphical Statistics* 34(3), 859-871).

AddiVortes replaces the decision trees of BART (Chipman, George and McCulloch,
2010) with Voronoi tessellations, so a cell is a region of the covariate space
rather than a box. The sampler is the Gibbs sampler of the paper.

## Installation

The package builds the core crate from source, so a Rust toolchain (`cargo`,
`rustc` >= 1.83) is required until wheels are published.

```sh
pip install -e python/
```

The `sklearn` and `arviz` extras add the scikit-learn estimators and the arviz
conversion.

## A first fit

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model

rng = np.random.default_rng(0)
x = rng.uniform(size=(200, 3))
y = 3.0 * (x[:, 0] - 0.4) ** 2 + 0.5 * x[:, 1] + rng.normal(scale=0.1, size=200)

fitted = Model(m=50, burn_in=100, draws=200).fit(x, y, random_state=1)

print("draws:", fitted.n_draws)
print("in-sample RMSE:", round(fitted.in_sample_rmse, 4))
print("prediction shape:", fitted.predict(x).shape)
```

`Model` holds the configuration and `fit` returns a
[`FittedModel`](api.md#thiessen.FittedModel), which answers prediction and
posterior queries.

## Intervals

`credible_interval` covers the mean function; `prediction_interval` covers a
new observation.

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model

rng = np.random.default_rng(0)
x = rng.uniform(size=(120, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=120)
fitted = Model(m=25, burn_in=50, draws=100).fit(x, y, random_state=1)

credible = fitted.credible_interval(x, level=0.9)
predictive = fitted.prediction_interval(x, level=0.9)
print("mean interval width:", round(float(np.mean(credible[:, 1] - credible[:, 0])), 4))
print(
    "predictive width:   ",
    round(float(np.mean(predictive[:, 1] - predictive[:, 0])), 4),
)
```

## Which covariates matter

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model

rng = np.random.default_rng(0)
x = rng.uniform(size=(150, 4))
y = 2.0 * x[:, 0] + rng.normal(scale=0.1, size=150)
fitted = Model(m=25, burn_in=50, draws=100).fit(x, y, random_state=1)

proportions = fitted.variable_inclusion_proportions()
print("inclusion proportions:", np.round(proportions, 3))
```

The proportions sum to one and give the share of active tessellation
dimensions falling on each covariate (Chipman, George and McCulloch, 2010,
s. 5.1).

## Saving a fit

`save` and `load` carry the core's representation, so a reloaded model
reproduces its predictions exactly.

```python exec="on" source="above" result="text"
import tempfile
from pathlib import Path

import numpy as np
from thiessen import FittedModel, Model

rng = np.random.default_rng(0)
x = rng.uniform(size=(60, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=60)
fitted = Model(m=10, burn_in=20, draws=40).fit(x, y, random_state=1)

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "fit.json"
    fitted.save(path)
    reloaded = FittedModel.load(path)

print("identical:", bool(np.array_equal(reloaded.predict(x), fitted.predict(x))))
```
