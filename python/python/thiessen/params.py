"""The parameter groups of a configuration.

`TermParams` describes one ensemble; ``mean_params`` and
``variance_params`` each take one. Geometry and structure nest inside it
the way the core's configuration nests them, so every name here is a name
in the stored configuration. The objects implement ``get_params`` and
``set_params``, so `sklearn.model_selection.GridSearchCV` routes into them
with keys of the form ``mean_params__tessellations``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

from ._params import Params, _dense

__all__ = ["GeometryParams", "StructureParams", "TermParams"]

MetricEntry = Union[str, "dict[str, dict[str, int]]"]


class GeometryParams(Params):
    """The covariate space of one ensemble.

    Parameters
    ----------
    metric : sequence, optional
        The metric of each covariate column, one entry per column in
        column order: ``'euclidean'``, ``'categorical'``, or
        ``{'spherical': {'sphere': k}}`` with `k` the sphere label. `None`
        is Euclidean on every column. Non-Euclidean columns are not
        scaled.
    sigma_c : float, default=0.8
        Centre-coordinate prior and proposal standard deviation sigma_c
        in the scaled space.
    """

    def __init__(
        self,
        metric: Sequence[MetricEntry] | None = None,
        sigma_c: float = 0.8,
    ) -> None:
        self.metric = metric
        self.sigma_c = sigma_c

    def _core(self) -> dict[str, Any]:
        return _dense({"metric": self.metric, "sigma_c": self.sigma_c})


class StructureParams(Params):
    """The covariate-inclusion prior of one ensemble.

    Parameters
    ----------
    omega : float, optional
        Dimension-count prior parameter omega; omega / p is the prior
        probability of including a covariate. `None` resolves to
        min(3, p) at fit. Must satisfy 0 < omega <= p.
    """

    def __init__(self, omega: float | None = None) -> None:
        self.omega = omega

    def _core(self) -> dict[str, Any]:
        return _dense({"omega": self.omega})


class TermParams(Params):
    """One ensemble of tessellations: its size, priors and covariate space.

    Parameters
    ----------
    tessellations : int, optional
        Number of tessellations in the ensemble. `None` resolves at fit
        to 200 as ``mean_params`` and to 0 as ``variance_params``; a
        positive count as ``variance_params`` selects the heteroscedastic
        model (the paper's count is 40).
    k : float, default=3.0
        Cell-value prior spread k: sigma_mu = w / (k sqrt(m)) with the
        half-width w the outcome family owns (Chipman, George and
        McCulloch, 2010, s. 4). The variance ensemble's inverse-gamma
        cells do not use it.
    lambda_c : float, default=5.0
        Cell-count prior rate lambda_c: b - 1 ~ Poisson(lambda_c). The
        default follows AddiVortes >= 0.6.8; Stone and Gosling (2025),
        s. 2.3, report 25.
    geometry : GeometryParams, optional
        The covariate space. `None` takes the defaults. The ensembles
        share one covariate space: set it on ``mean_params`` and it
        applies to ``variance_params`` as well.
    structure : StructureParams, optional
        The covariate-inclusion prior. `None` takes the defaults. Shared
        between the ensembles like `geometry`.

    Examples
    --------
    >>> from thiessen import TermParams
    >>> TermParams(tessellations=200, k=3.0)
    TermParams(tessellations=200)
    """

    def __init__(
        self,
        tessellations: int | None = None,
        k: float = 3.0,
        lambda_c: float = 5.0,
        geometry: GeometryParams | None = None,
        structure: StructureParams | None = None,
    ) -> None:
        self.tessellations = tessellations
        self.k = k
        self.lambda_c = lambda_c
        self.geometry = geometry
        self.structure = structure

    def _core(self) -> dict[str, Any]:
        group = _dense(
            {
                "tessellations": self.tessellations,
                "k": self.k,
                "lambda_c": self.lambda_c,
            }
        )
        if self.geometry is not None:
            if not isinstance(self.geometry, GeometryParams):
                raise TypeError(
                    "geometry must be GeometryParams(...) or None; "
                    f"got {self.geometry!r}"
                )
            group["geometry"] = self.geometry._core()
        if self.structure is not None:
            if not isinstance(self.structure, StructureParams):
                raise TypeError(
                    "structure must be StructureParams(...) or None; "
                    f"got {self.structure!r}"
                )
            structure = self.structure._core()
            if structure:
                group["structure"] = structure
        return group
