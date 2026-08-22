"""Render docs/parity.md from the core's configuration spec.

The table maps every option of the core's configuration format to its
place on the Python and R surfaces. It is rendered from the core's own
serialised defaults, never edited by hand. The Python suite renders it
again and fails on any difference, and each binding's parity test proves
every listed option constructible, so drift between the table, the core
and the surfaces is a test failure.

Usage: python tools/parity_table.py [--write], with the thiessen Python
package importable. Without --write the table is printed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from thiessen import _native

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "parity.md"

#: Groups whose serialised form has no field on the stable surface.
UNEXPOSED = {
    "cell": "no stable field; the within-cell basis is experimental and core-only"
}

HEADER = """\
# Binding parity

The core's configuration format against its place on each binding's
surface. Rendered by `tools/parity_table.py` from the core's serialised
defaults. The Python suite renders it again and fails on any difference,
and each binding's parity test proves every listed option constructible,
so this file is regenerated, never edited.

Every name is identical across the three surfaces. Python groups run-length
settings flat on the estimator and `Model`; R groups them in
`general_params()`. The one R shortcut, `thiessen_control(tessellations =)`,
sets `mean_params.tessellations`.

| Core option | Python | R |
| --- | --- | --- |
"""

FOOTER = """
Groups without a row: `mean_params.cell` and `variance_params.cell` carry
no field on the stable surface; the within-cell basis is experimental and
core-only (`docs/experimental.md`).

The seed is not part of the configuration: it is `random_state` in Python
and `seed` in R, resolved by each language's rule and passed to the core
unchanged.
"""


def _python_location(group: str, path: list[str]) -> str:
    if group == "general_params":
        return f"`{path[0]}=` on `Model` and the estimators"
    if path[0] == "geometry":
        return f"`GeometryParams({path[1]}=)` in `{group}=`"
    if path[0] == "structure":
        return f"`StructureParams({path[1]}=)` in `{group}=`"
    return f"`TermParams({path[0]}=)` in `{group}=`"


def _r_location(group: str, path: list[str]) -> str:
    if group == "general_params":
        return f"`general_params({path[0]} = )`"
    if path[0] == "geometry":
        return f"`geometry_params({path[1]} = )` in `{group} = `"
    if path[0] == "structure":
        return f"`structure_params({path[1]} = )` in `{group} = `"
    return f"`term_params({path[0]} = )` in `{group} = `"


def _rows() -> list[str]:
    rows: list[str] = []
    for family in json.loads(_native.outcome_defaults()):
        (name, params) = next(iter(family.items()))
        for argument in params:
            rows.append(
                f"| `outcome.{name}.{argument}` "
                f"| `{name}({argument}=)` in `outcome=` "
                f"| `{name}({argument} = )` in `outcome = ` |"
            )
    config = json.loads(_native.default_config())
    for group in ("mean_params", "variance_params", "general_params"):
        for path in _leaves(config[group]):
            if path[0] in UNEXPOSED:
                continue
            core = f"{group}.{'.'.join(path)}"
            rows.append(
                f"| `{core}` | {_python_location(group, path)} "
                f"| {_r_location(group, path)} |"
            )
    return rows


def _leaves(tree: dict, prefix: list[str] | None = None) -> list[list[str]]:
    prefix = prefix or []
    paths: list[list[str]] = []
    for key, value in tree.items():
        if isinstance(value, dict):
            if not value:
                paths.append([*prefix, key])
            else:
                paths.extend(_leaves(value, [*prefix, key]))
        else:
            paths.append([*prefix, key])
    return paths


def render() -> str:
    """Return the parity table as markdown."""
    return HEADER + "\n".join(_rows()) + "\n" + FOOTER


def main() -> int:
    """Print the table, or write it with --write."""
    text = render()
    if "--write" in sys.argv[1:]:
        TARGET.write_text(text)
        print(f"wrote {TARGET}")
        return 0
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
