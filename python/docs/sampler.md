# The sampler API

!!! note "Experimental"
    This interface may change in a minor release without a deprecation
    cycle. The models it drives and the draws it produces carry the same
    guarantees as `Model.fit`.

`thiessen.sampler.Sampler` is the researcher's interface, after the
updatable sampler object of dbarts and the low-level interface of
stochtree: construct with the configuration, the data and a seed, then
drive the Gibbs loop yourself. Burn-in and thinning are the caller's loop,
and the response may be replaced between sweeps, which is what makes
censored responses, imputation between sweeps, custom likelihoods through
the response, and prototyping of outcome models possible. Anything that is
not an outcome family or a setting goes through this loop.

The response is on the caller's scale through an affine map frozen at
construction, so a response outside the training range is legitimate. The
sampler owns its RNG, seeded at construction with the chain-0 seed of
`fit`, so driving the configured schedule by hand reproduces `fit` bit for
bit; the loop cannot rewire tessellation membership or cell internals.

## A fit as its own loop

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import Model, TermParams
from thiessen.sampler import Sampler

rng = np.random.default_rng(0)
x = rng.uniform(size=(60, 2))
y = x[:, 0] + rng.normal(scale=0.1, size=60)

sampler = Sampler(x, y, mean_params=TermParams(tessellations=10), random_state=1)
sampler.step(20)
for _ in range(40):
    sampler.step(1)
    sampler.keep()
driven = sampler.finish()

model = Model(mean_params=TermParams(tessellations=10), burn_in=20, draws=40)
fitted = model.fit(x, y, random_state=1)
print("matches fit:", bool(np.array_equal(driven.predict(x), fitted.predict(x))))
```

## Imputation between sweeps

A right-censored response can be redrawn from its conditional before each
sweep: read the current mean function and noise level, redraw the censored
values, and hand the response back.

```python exec="on" source="above" result="text"
import numpy as np
from thiessen import TermParams
from thiessen.sampler import Sampler

rng = np.random.default_rng(0)
x = rng.uniform(size=(60, 2))
latent = x[:, 0] + rng.normal(scale=0.1, size=60)
limit = np.quantile(latent, 0.8)
censored = latent > limit
y = np.minimum(latent, limit)

sampler = Sampler(x, y, mean_params=TermParams(tessellations=10), random_state=1)
sampler.step(20)
for _ in range(40):
    mean = sampler.fitted_values()
    scale = np.sqrt(sampler.noise_variances())
    working = y.copy()
    # A draw from the tail above the limit, by inversion.
    tail = 1.0 - rng.uniform(size=censored.sum())
    working[censored] = limit + scale[censored] * tail
    sampler.set_response(working)
    sampler.step(1)
    sampler.keep()

fitted = sampler.finish()
print("draws:", fitted.n_draws)
```

The redraw above is illustrative; a faithful truncated-normal draw would
invert the conditional tail probability.
