"""Encoding of declared categorical columns into the core's design.

A categorical covariate reaches the core either as d - 1 indicator columns
under the Euclidean metric, the encoding of `model.matrix` treatment
contrasts and of upstream AddiVortes, or as one column of integer level codes
under the Eskin metric. Which one applies follows the column's entry in
`metric`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Union

import numpy as np
import numpy.typing as npt

__all__ = ["Encoding", "columns_of", "resolve_mask"]

MetricEntry = Union[str, "dict[str, dict[str, int]]"]


def columns_of(x: Any) -> list[npt.NDArray[Any]]:
    """Split `x` into one array per column, preserving non-numeric dtypes.

    Parameters
    ----------
    x : array_like or pandas.DataFrame
        The design.

    Returns
    -------
    list of numpy.ndarray
        One array per column, in column order.
    """
    if hasattr(x, "iloc") and hasattr(x, "columns"):
        return [np.asarray(x.iloc[:, j]) for j in range(x.shape[1])]
    array = np.asarray(x)
    if array.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got {array.ndim} dimensions")
    return [array[:, j] for j in range(array.shape[1])]


def _dtype_mask(x: Any, n_features: int) -> npt.NDArray[np.bool_]:
    """The columns a pandas categorical dtype declares."""
    dtypes = getattr(x, "dtypes", None)
    if dtypes is None:
        raise ValueError(
            "categorical_features='from_dtype' needs a data frame with "
            "categorical dtypes"
        )
    return np.array([str(dtype) == "category" for dtype in dtypes], dtype=bool)


def resolve_mask(
    categorical_features: str | Sequence[Any] | None,
    x: Any,
    n_features: int,
) -> npt.NDArray[np.bool_]:
    """Resolve `categorical_features` to a boolean mask over the columns.

    Parameters
    ----------
    categorical_features : None, 'from_dtype', or array_like
        `None` takes every column as numeric; 'from_dtype' reads the pandas
        categorical dtypes; an array of column indices or a boolean mask
        names the columns.
    x : array_like or pandas.DataFrame
        The design, needed for 'from_dtype'.
    n_features : int
        The number of columns.

    Returns
    -------
    numpy.ndarray of bool
        One entry per column.

    Raises
    ------
    ValueError
        For an unknown string, a mask of the wrong length, or an index
        outside the columns.
    """
    if categorical_features is None:
        return np.zeros(n_features, dtype=bool)
    if isinstance(categorical_features, str):
        if categorical_features != "from_dtype":
            raise ValueError(
                "categorical_features must be None, 'from_dtype', or an array "
                f"of indices or a boolean mask, got {categorical_features!r}"
            )
        return _dtype_mask(x, n_features)
    declared = np.asarray(categorical_features)
    if declared.dtype == bool:
        if declared.shape != (n_features,):
            raise ValueError(
                f"a boolean categorical_features mask must have {n_features} "
                f"entries, got {declared.shape[0]}"
            )
        return declared
    mask = np.zeros(n_features, dtype=bool)
    for index in declared.astype(int, copy=False).ravel():
        if not 0 <= index < n_features:
            raise ValueError(
                f"categorical_features index {index} is outside the "
                f"{n_features} columns"
            )
        mask[index] = True
    return mask


def _entry(metric: Sequence[MetricEntry] | None, column: int) -> MetricEntry:
    if metric is None or column >= len(metric):
        return "euclidean"
    return metric[column]


class Encoding:
    """The caller's columns as the core's design.

    Attributes
    ----------
    mask : numpy.ndarray of bool
        The columns taken as categorical.
    levels : list
        The levels of each categorical column, `None` elsewhere.
    core_metric : list
        The metric of each column of the encoded design.
    """

    def __init__(
        self,
        mask: npt.NDArray[np.bool_],
        levels: list[npt.NDArray[Any] | None],
        core_metric: list[MetricEntry],
        codes: list[bool],
    ) -> None:
        self.mask = mask
        self.levels = levels
        self.core_metric = core_metric
        self._codes = codes

    @classmethod
    def fit(
        cls,
        columns: list[npt.NDArray[Any]],
        mask: npt.NDArray[np.bool_],
        metric: Sequence[MetricEntry] | None,
    ) -> Encoding:
        """Learn the levels of every declared categorical column.

        Parameters
        ----------
        columns : list of numpy.ndarray
            One array per input column.
        mask : numpy.ndarray of bool
            The columns taken as categorical.
        metric : sequence or None
            The metric of each input column.

        Returns
        -------
        Encoding
            The encoding to apply at fit and at predict.

        Raises
        ------
        ValueError
            If a categorical column holds one level, which carries no
            information and leaves the core a constant column.
        """
        levels: list[npt.NDArray[Any] | None] = []
        core_metric: list[MetricEntry] = []
        codes: list[bool] = []
        for index, column in enumerate(columns):
            entry = _entry(metric, index)
            if not mask[index]:
                levels.append(None)
                codes.append(False)
                core_metric.append(entry)
                continue
            distinct = np.unique(column)
            if distinct.size < 2:
                raise ValueError(
                    f"categorical column {index} has {distinct.size} level(s); "
                    "at least two are needed"
                )
            levels.append(distinct)
            as_codes = entry == "categorical"
            codes.append(as_codes)
            if as_codes:
                core_metric.append("categorical")
            else:
                core_metric.extend(["euclidean"] * (distinct.size - 1))
        return cls(mask, levels, core_metric, codes)

    def transform(self, columns: list[npt.NDArray[Any]]) -> npt.NDArray[np.float64]:
        """Apply the encoding.

        Parameters
        ----------
        columns : list of numpy.ndarray
            One array per input column.

        Returns
        -------
        numpy.ndarray
            The encoded design, float64.

        Raises
        ------
        ValueError
            For a level absent from the fit.
        """
        out: list[npt.NDArray[np.float64]] = []
        for index, column in enumerate(columns):
            known = self.levels[index]
            if known is None:
                out.append(np.asarray(column, dtype=np.float64))
                continue
            position = np.searchsorted(known, column)
            position = np.clip(position, 0, known.size - 1)
            if not np.all(known[position] == column):
                unseen = np.asarray(column)[known[position] != column][0]
                raise ValueError(
                    f"column {index} holds level {unseen!r}, which is not one "
                    "of the levels of the fit"
                )
            if self._codes[index]:
                out.append(position.astype(np.float64))
                continue
            for level in range(1, known.size):
                out.append((position == level).astype(np.float64))
        return np.column_stack(out) if out else np.empty((0, 0), dtype=np.float64)
