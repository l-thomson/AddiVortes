"""The configuration fields and their JSON encoding for the core."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["FIELDS", "_config_json", "_flat_config"]

#: The configuration fields, in the order of the core's `Config`.
FIELDS = (
    "model",
    "m",
    "nu",
    "q",
    "k",
    "sigma_c",
    "omega",
    "lambda_c",
    "burn_in",
    "draws",
    "thinning",
    "prior_only",
    "offset",
    "m_var",
    "metric",
)


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


def _config_json(params: Mapping[str, Any]) -> str:
    """Encode the set configuration fields as the core's JSON.

    Fields left as `None` are omitted, so the core's default applies. The
    core rejects unknown fields and validates every value, so no validation
    is repeated here.

    Parameters
    ----------
    params : mapping
        Field name to value, `None` for unset.

    Returns
    -------
    str
        The configuration as JSON.
    """
    flat: dict[str, Any] = {
        name: _plain(value) for name, value in params.items() if value is not None
    }
    return json.dumps(_grouped(flat))


def _grouped(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Arrange the flat fields into the core's outcome and parameter groups."""
    flat = dict(flat)
    model = flat.pop("model", "gaussian") or "gaussian"
    heteroscedastic = model == "heteroscedastic"

    if model in ("gaussian", "heteroscedastic"):
        outcome_kind = "gaussian"
    else:
        # "probit", or an unknown name the core rejects as an outcome.
        outcome_kind = model
    outcome: dict[str, Any] = {}
    for name in ("nu", "q"):
        value = flat.pop(name, None)
        if value is not None and outcome_kind == "gaussian":
            outcome[name] = value
    offset = flat.pop("offset", None)
    if offset is not None and outcome_kind == "probit":
        outcome["offset"] = offset

    term: dict[str, Any] = {}
    for name in ("k", "lambda_c"):
        value = flat.pop(name, None)
        if value is not None:
            term[name] = value
    geometry: dict[str, Any] = {}
    sigma_c = flat.pop("sigma_c", None)
    if sigma_c is not None:
        geometry["sigma_c"] = sigma_c
    metric = flat.pop("metric", None)
    if metric is not None:
        geometry["metric"] = metric
    if geometry:
        term["geometry"] = geometry
    omega = flat.pop("omega", None)
    if omega is not None:
        term["structure"] = {"omega": omega}

    mean = dict(term)
    m = flat.pop("m", None)
    if m is not None:
        mean["num_tessellations"] = m

    # The slots share geometry and structure; the ensemble count is the
    # variance slot's own.
    m_var = flat.pop("m_var", None)
    variance = {
        key: value for key, value in term.items() if key in ("geometry", "structure")
    }
    if heteroscedastic:
        variance["num_tessellations"] = m_var if m_var is not None else 40

    general = {
        name: flat.pop(name)
        for name in ("burn_in", "draws", "thinning", "prior_only")
        if flat.get(name) is not None
    }

    grouped: dict[str, Any] = {"outcome": {outcome_kind: outcome}}
    if mean:
        grouped["mean_params"] = mean
    if variance:
        grouped["variance_params"] = variance
    if general:
        grouped["general_params"] = general
    # Anything else passes through for the core to reject by name.
    grouped.update(flat)
    return grouped


def _flat_config(grouped: Mapping[str, Any]) -> dict[str, Any]:
    """Return the flat fields of a grouped configuration, in FIELDS order."""
    outcome = grouped.get("outcome", {})
    kind, params = next(iter(outcome.items())) if outcome else ("gaussian", {})
    mean = grouped.get("mean_params", {})
    variance = grouped.get("variance_params", {})
    general = grouped.get("general_params", {})
    geometry = mean.get("geometry", {})
    structure = mean.get("structure", {})
    m_var = variance.get("num_tessellations") or 0
    return {
        "model": "heteroscedastic" if kind == "gaussian" and m_var > 0 else kind,
        "m": mean.get("num_tessellations", 200) or 200,
        "nu": params.get("nu", 6.0) if kind == "gaussian" else 6.0,
        "q": params.get("q", 0.85) if kind == "gaussian" else 0.85,
        "k": mean.get("k"),
        "sigma_c": geometry.get("sigma_c"),
        "omega": structure.get("omega"),
        "lambda_c": mean.get("lambda_c"),
        "burn_in": general.get("burn_in"),
        "draws": general.get("draws"),
        "thinning": general.get("thinning"),
        "prior_only": general.get("prior_only"),
        "offset": params.get("offset") if kind == "probit" else None,
        "m_var": m_var if m_var > 0 else 40,
        "metric": geometry.get("metric"),
    }
