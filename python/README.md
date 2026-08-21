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

## Documentation

The reproducibility contract, the input-data contract and the testing
strategy are in the repository: `README.md`, `docs/models.md` and
`docs/testing.md`.

## Licence

MIT or Apache-2.0, at your option.
