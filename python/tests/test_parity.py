"""Parity with the core's configuration spec, in both directions.

The core's serialised defaults are the authority: every option there must
be reachable from the surface, every surface argument must serialise to a
core option, and the parity table must be exactly what the generator
renders from the spec.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from thiessen import (
    GeometryParams,
    Model,
    StructureParams,
    TermParams,
    _native,
    gaussian,
    probit,
)
from thiessen.estimators import AddiVortesRegressor

ROOT = Path(__file__).resolve().parents[2]

#: Serialised groups with no field on the stable surface.
UNEXPOSED = {"mean_params.cell", "variance_params.cell"}


def _full_term() -> TermParams:
    return TermParams(
        tessellations=8,
        k=2.0,
        lambda_c=4.0,
        geometry=GeometryParams(metric=["euclidean", "euclidean"], sigma_c=0.7),
        structure=StructureParams(omega=1.5),
    )


def _full_model() -> Model:
    return Model(
        outcome=gaussian(nu=5.0, q=0.9),
        mean_params=_full_term(),
        variance_params=_full_term(),
        burn_in=5,
        draws=6,
        thinning=2,
        prior_only=True,
    )


def _paths(tree: dict[str, Any], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths |= _paths(value, path + ".")
        else:
            paths.add(path)
    return paths


def test_every_surface_argument_is_a_core_option():
    """The core accepts the fully populated surface: no silent extras."""
    _native.validate_config(_full_model()._json())
    _native.validate_config(Model(outcome=probit(offset=0.5))._json())


def test_every_core_option_is_reachable_from_the_surface():
    core = _paths(json.loads(_native.default_config()))
    surface = _paths(json.loads(_full_model()._json()))

    missing = {path for path in core - surface if path not in UNEXPOSED}
    assert not missing, f"core options unreachable from the surface: {missing}"


def test_every_outcome_family_option_is_a_constructor_argument():
    import inspect

    constructors: dict[str, Any] = {"gaussian": gaussian, "probit": probit}
    for family in json.loads(_native.outcome_defaults()):
        (name, params) = next(iter(family.items()))
        arguments = set(inspect.signature(constructors[name]).parameters)
        assert set(params) == arguments, name


@pytest.mark.skipif(
    not (ROOT / "docs" / "parity.md").is_file(), reason="not in the source tree"
)
def test_the_parity_table_is_what_the_generator_renders():
    spec = importlib.util.spec_from_file_location(
        "parity_table", ROOT / "tools" / "parity_table.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    committed = (ROOT / "docs" / "parity.md").read_text()
    assert module.render() == committed, (
        "docs/parity.md drifted from the core spec; regenerate with "
        "python tools/parity_table.py --write"
    )


def test_the_factor_encoding_is_the_shared_fixture():
    """Nine rows, three levels: the encoding the R suite asserts as well."""
    n = 9
    x = np.column_stack([np.arange(n) / (n - 1), np.arange(n) % 3.0])
    y = x[:, 0] + 0.5 * x[:, 1]

    model = AddiVortesRegressor(
        categorical_features=[1],
        mean_params=TermParams(tessellations=4),
        burn_in=2,
        draws=2,
        random_state=1,
    ).fit(x, y)

    encoded = model._encoding.transform([x[:, 0], x[:, 1]])
    expected = np.column_stack(
        [
            x[:, 0],
            (x[:, 1] == 1.0).astype(np.float64),
            (x[:, 1] == 2.0).astype(np.float64),
        ]
    )
    np.testing.assert_array_equal(encoded, expected)
