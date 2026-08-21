"""Coercion of the design and the response to the layout the core takes."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["_as_design", "_as_response"]


def _as_design(x: Any) -> npt.NDArray[np.float64]:
    """Coerce `x` to a two-dimensional C-contiguous float64 array.

    Parameters
    ----------
    x : array_like
        The design, one row per observation.

    Returns
    -------
    numpy.ndarray
        The design.

    Raises
    ------
    ValueError
        If `x` is not two-dimensional.
    """
    design = np.ascontiguousarray(x, dtype=np.float64)
    if design.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got {design.ndim} dimensions")
    return design


def _as_response(y: Any) -> npt.NDArray[np.float64]:
    """Coerce `y` to a one-dimensional C-contiguous float64 array.

    Parameters
    ----------
    y : array_like
        The response, one value per observation. A column of shape (n, 1) is
        accepted and flattened.

    Returns
    -------
    numpy.ndarray
        The response.

    Raises
    ------
    ValueError
        If `y` is not one-dimensional after flattening a single column.
    """
    response = np.ascontiguousarray(y, dtype=np.float64)
    if response.ndim == 2 and response.shape[1] == 1:
        response = response.reshape(-1)
    if response.ndim != 1:
        raise ValueError(f"y must be one-dimensional, got shape {response.shape}")
    return response
