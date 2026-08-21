"""Bayesian additive Voronoi tessellations (AddiVortes).

An implementation of Stone and Gosling (2025), *Journal of Computational and
Graphical Statistics* 34(3), 859-871: Bayesian regression on a sum of
Voronoi tessellations, with the Gaussian, probit and heteroscedastic models.

The reproducibility contract, the input-data contract and the testing
strategy are those of the core crate: the same seed, package version and
target give identical draws, and draws do not depend on thread count.
"""

from __future__ import annotations

from importlib import metadata as _metadata

from ._native import CORE_VERSION, ThiessenError
from .model import FittedModel, Model

__all__ = [
    "CORE_VERSION",
    "FittedModel",
    "Model",
    "ThiessenError",
    "__version__",
]

try:
    __version__: str = _metadata.version("thiessen")
except _metadata.PackageNotFoundError:  # not installed, as in a source tree
    __version__ = CORE_VERSION
