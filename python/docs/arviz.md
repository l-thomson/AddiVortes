# arviz

`FittedModel.to_inference_data(X, y)` returns the arviz `DataTree` of a fit,
with the groups of the PyMC and numpyro convention. It needs the `arviz` extra.

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model

rng = np.random.default_rng(0)
x = rng.uniform(size=(120, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=120)

fitted = Model(m=25, burn_in=50, draws=100).fit(x, y, random_state=1)
data = fitted.to_inference_data(x, y)
print("groups:", sorted(data.children))
print("posterior:", sorted(data["posterior"].dataset.data_vars))
```

| Group | Variables |
| --- | --- |
| `posterior` | `mu`, the mean function per draw; `sigma` under the Gaussian model; `cell_count`; `dimension_count` |
| `posterior_predictive` | `y`, one replicate per draw |
| `log_likelihood` | `y`, pointwise per draw |
| `observed_data` | `y` |

The sampler runs one chain, so every group has a chain dimension of one. The
observation dimension is labelled, so the arviz summaries and the
information-criterion functions work directly.

```python exec="on" source="above" result="text"
import arviz as az
import numpy as np
from thiessen import Model

rng = np.random.default_rng(0)
x = rng.uniform(size=(120, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=120)

data = (
    Model(m=25, burn_in=50, draws=100).fit(x, y, random_state=1).to_inference_data(x, y)
)
print(az.summary(data, var_names=["sigma", "cell_count"]))
```

`sigma` appears under the Gaussian model alone: the probit model's latent
variance is one and the heteroscedastic model's varies with x, so neither has a
scalar to report. Use `predict_variance` for the heteroscedastic variance.

The predictive replicates are drawn in numpy from the fit's resolved seed, so
they are reproducible without being draws of the core. Everything else in the
tree comes from the kept draws.
