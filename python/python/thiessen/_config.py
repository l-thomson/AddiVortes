"""The configuration fields and their JSON encoding for the core."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["FIELDS", "_config_json"]

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
    encoded: dict[str, Any] = {
        name: _plain(value) for name, value in params.items() if value is not None
    }
    return json.dumps(encoded)
