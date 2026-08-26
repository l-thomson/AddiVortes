"""Building the core's JSON configuration from the surface objects.

The groups serialise exactly as the user set them; fields left unset are
omitted, so the core's defaults apply, and the core rejects unknown fields
and validates every value, so no validation is repeated here.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._params import Outcome
from .params import MetricEntry, TermParams, _metric_entry

__all__ = ["_config_json"]


def _plain(value: Any) -> Any:
    """Convert numpy scalars and sequences to JSON-representable values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _config_json(
    outcome: Outcome | None,
    mean_params: TermParams | None,
    variance_params: TermParams | None,
    burn_in: int,
    draws: int,
    thinning: int,
    prior_only: bool,
    core_metric: Sequence[MetricEntry] | None = None,
) -> str:
    """Encode the configuration as the core's JSON.

    Parameters
    ----------
    outcome : Outcome or None
        The outcome family; `None` takes the core's default.
    mean_params, variance_params : TermParams or None
        The two ensembles; `None` takes the core's defaults.
    burn_in, draws, thinning : int
        The sweep schedule.
    prior_only : bool
        Sample from the prior.
    core_metric : sequence, optional
        A metric over the encoded design that replaces the metric of both
        ensembles, used where a categorical encoding has changed the
        column count.

    Returns
    -------
    str
        The configuration as JSON.

    Raises
    ------
    TypeError
        For a group that is not its parameter object.
    """
    if outcome is not None and not isinstance(outcome, Outcome):
        raise TypeError(
            "outcome must be a family object from one of the family "
            f"constructors, gaussian(), probit() and the rest; got {outcome!r}"
        )
    for name, group in (
        ("mean_params", mean_params),
        ("variance_params", variance_params),
    ):
        if group is not None and not isinstance(group, TermParams):
            raise TypeError(f"{name} must be TermParams(...) or None; got {group!r}")

    mean = mean_params._core() if mean_params is not None else {}
    variance = variance_params._core() if variance_params is not None else {}
    if core_metric is not None:
        mean.setdefault("geometry", {})["metric"] = [
            _metric_entry(entry) for entry in core_metric
        ]
    # The ensembles share one covariate space; the core requires the slots
    # to declare it identically while per-ensemble geometry awaits its
    # identification argument.
    if variance:
        for shared in ("geometry", "structure"):
            if shared not in variance and shared in mean:
                variance[shared] = mean[shared]

    grouped: dict[str, Any] = {
        "general_params": {
            "burn_in": burn_in,
            "draws": draws,
            "thinning": thinning,
            "prior_only": prior_only,
        }
    }
    if outcome is not None:
        grouped["outcome"] = outcome._core()
    if mean:
        grouped["mean_params"] = mean
    if variance:
        grouped["variance_params"] = variance
    return json.dumps(_plain(grouped))
