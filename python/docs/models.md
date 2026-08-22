# Models

The outcome family selects the observation model: `gaussian()` or
`probit()`, with the heteroscedastic extension of the Gaussian family
selected by attaching `variance_params` with a positive tessellation count.
The three are the published method: Stone and Gosling (2025) and CRAN
AddiVortes.

| Model | Selected by | Response | `sigma()` | `predict` returns | `predict_variance` |
| --- | --- | --- | --- | --- | --- |
| gaussian | `gaussian()` | continuous | one sigma per draw | E[y \| x] | sigma^2, constant in x |
| probit | `probit()` | labels in {0, 1} | empty | P(y = 1 \| x) | not applicable |
| heteroscedastic | `gaussian()` + `variance_params` | continuous | empty | E[y \| x] | s^2(x), varying in x |

## Gaussian

y = f(x) + e with e ~ N(0, sigma^2) and f a sum of m tessellations. The sweep
draws sigma^2 from its inverse-gamma conditional.

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(120, 2))
y = x[:, 0] ** 2 + rng.normal(scale=0.1, size=120)

fitted = Model(mean_params=TermParams(tessellations=25), burn_in=50, draws=100).fit(
    x, y, random_state=1
)
print("sigma draws:", fitted.sigma().shape)
print("posterior mean sigma:", round(float(fitted.sigma().mean()), 4))
```

## Probit

P(y = 1 | x) = Phi(c + f(x)), with the Albert and Chib (1993) latent
augmentation and unit latent variance. The offset c defaults to Phi^-1(ybar),
the BART `binaryOffset` default, and is resolved at fit. `predict` gives the
posterior mean probability; `predict_latent` gives c + f(x).

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams, probit

rng = np.random.default_rng(0)
x = rng.uniform(size=(150, 2))
y = (x[:, 0] > 0.5).astype(float)

model = Model(
    outcome=probit(),
    mean_params=TermParams(tessellations=25),
    burn_in=50,
    draws=100,
)
fitted = model.fit(x, y, random_state=1)
probabilities = fitted.predict(x)
print(
    "range:", round(float(probabilities.min()), 3), round(float(probabilities.max()), 3)
)
print("latent draws:", fitted.predict_latent(x).shape)
```

`sigma()` is empty, as the latent variance is one. `predict_variance` and
`prediction_interval` raise, as a two-point distribution has neither.

## Heteroscedastic

y = f(x) + e with e ~ N(0, s^2(x)) and s^2 a multiplicative ensemble of
inverse-gamma variance tessellations, sized by the `variance_params`
tessellation count. `predict_variance` gives s^2(x) per draw.

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams

rng = np.random.default_rng(0)
x = rng.uniform(size=(200, 2))
y = x[:, 0] + rng.normal(scale=0.05 + 0.3 * x[:, 0], size=200)

model = Model(
    mean_params=TermParams(tessellations=25),
    variance_params=TermParams(tessellations=20),
    burn_in=50,
    draws=100,
)
fitted = model.fit(x, y, random_state=1)
variance = fitted.predict_variance(x).mean(axis=0)
order = np.argsort(x[:, 0])
print("variance at low x: ", round(float(variance[order[:20]].mean()), 4))
print("variance at high x:", round(float(variance[order[-20:]].mean()), 4))
```

## Column metrics

The geometry's `metric` (`TermParams(geometry=GeometryParams(metric=[...]))`)
sets the distance of each covariate column, one entry per column in column
order. Non-Euclidean columns are not scaled.

| Entry | Squared distance | Centre coordinates |
| --- | --- | --- |
| `'euclidean'` | Euclidean on the column scaled to [-0.5, 0.5] | N(0, sigma_c^2) |
| `{'spherical': {'sphere': k}}` | great-circle angle, radians | N(mid, sd^2), longitude wrapped |
| `'categorical'` | 2 / n^2 per mismatch, n levels | uniform over the levels |

Columns sharing a `sphere` label form one sphere: its latitudes, then its
longitude last. The categorical weight is that of Eskin et al. (2002), CRAN
AddiVortes `metric = "C"`.

## The stable surface

These three models are the published method and follow semantic versioning.
Everything else the core crate adds sits behind its `experimental` Cargo
feature, which this package does not enable, so a configuration or a saved
model naming such an option is rejected with the core's message naming the
feature. The table of experimental items and their status is
`docs/experimental.md` in the repository. A graduated item is exposed here as
any other option, with no separate opt-in.

## References

Albert, J. H. and Chib, S. (1993). Bayesian analysis of binary and
polychotomous response data. *Journal of the American Statistical
Association*, 88(422), 669-679.

Eskin, E., Arnold, A., Prerau, M., Portnoy, L. and Stolfo, S. (2002). A
geometric framework for unsupervised anomaly detection. In *Applications of
Data Mining in Computer Security*, 77-101.

Stone, E. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive Voronoi
tessellations. *Journal of Computational and Graphical Statistics*, 34(3),
859-871.
