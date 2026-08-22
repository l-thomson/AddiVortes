"""Bayesian additive Voronoi tessellations (AddiVortes).

An implementation of Stone and Gosling (2025), *Journal of Computational and
Graphical Statistics* 34(3), 859-871: Bayesian regression on a sum of
Voronoi tessellations, with the Gaussian, probit and heteroscedastic models.

The outcome model is a family object from `gaussian` or `probit`; the two
ensembles are `TermParams` groups. The names are those of the stored
configuration and of the R package.

The reproducibility contract, the input-data contract and the testing
strategy are those of the core crate: the same seed, package version and
target give identical draws, and draws do not depend on thread count.
"""

from __future__ import annotations

from importlib import metadata as _metadata

from ._native import CORE_VERSION, ThiessenError
from .families import gaussian, probit
from .model import FittedModel, Model
from .params import GeometryParams, StructureParams, TermParams

__all__ = [
    "CORE_VERSION",
    "FittedModel",
    "GeometryParams",
    "Model",
    "StructureParams",
    "TermParams",
    "ThiessenError",
    "__version__",
    "gaussian",
    "probit",
]

try:
    __version__: str = _metadata.version("thiessen")
except _metadata.PackageNotFoundError:  # not installed, as in a source tree
    __version__ = CORE_VERSION
