"""The parameter-object protocol of the families and parameter groups.

The objects store their constructor arguments verbatim and implement
``get_params`` and ``set_params``, the pattern of
`sklearn.gaussian_process.kernels.Kernel`, so `sklearn.base.clone` and
``<name>__<parameter>`` routing work while the package itself does not
import scikit-learn.
"""

from __future__ import annotations

import inspect
from typing import Any

__all__ = ["Params"]


def _is_default(value: Any, default: Any) -> bool:
    """Whether `value` equals its signature default."""
    if default is inspect.Parameter.empty:
        return False
    if value is default:
        return True
    try:
        return bool(value == default)
    except (TypeError, ValueError):
        return False


def _dense(fields: dict[str, Any]) -> dict[str, Any]:
    """Return `fields` without the entries left as `None`."""
    return {name: value for name, value in fields.items() if value is not None}


class Params:
    """Constructor arguments as parameters, with nested routing.

    Subclasses set every constructor argument as an attribute of the same
    name and change nothing else, so a copy constructed from `get_params`
    is the object.
    """

    def _param_names(self) -> tuple[str, ...]:
        signature = inspect.signature(type(self).__init__)
        return tuple(
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind
            in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY)
        )

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the parameters of this object.

        Parameters
        ----------
        deep : bool, default=True
            Also return the parameters of nested parameter objects, keyed
            ``<name>__<parameter>``.

        Returns
        -------
        dict
            Parameter name to value.
        """
        params: dict[str, Any] = {}
        for name in self._param_names():
            value = getattr(self, name)
            params[name] = value
            if deep and hasattr(value, "get_params"):
                for key, sub_value in value.get_params(deep=True).items():
                    params[f"{name}__{key}"] = sub_value
        return params

    def set_params(self, **params: Any) -> Params:
        """Set parameters, routing ``<name>__<parameter>`` to nested objects.

        Parameters
        ----------
        **params : dict
            Parameter name to value.

        Returns
        -------
        Params
            This object.

        Raises
        ------
        ValueError
            For a name that is not a parameter of this object.
        """
        if not params:
            return self
        valid = self._param_names()
        nested: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            name, delimiter, sub_key = key.partition("__")
            if name not in valid:
                raise ValueError(
                    f"Invalid parameter {name!r} for {self!r}. "
                    f"Valid parameters are: {sorted(valid)!r}."
                )
            if delimiter:
                nested.setdefault(name, {})[sub_key] = value
            else:
                setattr(self, name, value)
        for name, sub_params in nested.items():
            getattr(self, name).set_params(**sub_params)
        return self

    def _display_name(self) -> str:
        return type(self).__name__

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return False
        mine = self.get_params(deep=False)
        theirs = other.get_params(deep=False)
        return bool(mine == theirs)

    def __repr__(self) -> str:
        signature = inspect.signature(type(self).__init__)
        shown = ", ".join(
            f"{name}={getattr(self, name)!r}"
            for name in self._param_names()
            if not _is_default(getattr(self, name), signature.parameters[name].default)
        )
        return f"{self._display_name()}({shown})"
