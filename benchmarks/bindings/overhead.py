"""The Python binding's cases against the core's time on the same work.

Reads the JSON pytest-benchmark writes and the JSON the `overhead` binary
writes, and prints the absolute difference beside the ratio. A ratio on a
call of a few hundred microseconds carries no information, so both are
reported.

    pytest tests/benchmarks --benchmark-json=../target/bindings/python.json
    python benchmarks/bindings/overhead.py target/bindings

No measurement happens here: pytest-benchmark took the times and this
joins them to the core's.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: The core case each benchmark is read against. The core has no per-call
#: case: crossing the boundary once per sweep and once for all of them is
#: the same loop to it, so both sampler rows are read against `sweeps`.
CORE_CASE = {
    "fit": "fit",
    "predict": "predict",
    "sweeps": "sweeps",
}

SIZE = re.compile(r"^n(\d+)$|^p(\d+)$")


def core_seconds(core: dict, case: str, n: int, p: int) -> float:
    for entry in core["cases"]:
        if entry["case"] == case and entry["n"] == n and entry["p"] == p:
            return float(entry["seconds"])
    raise SystemExit(f"no core time for {case} at n={n} p={p}")


def parse(name: str) -> tuple[str, int, int]:
    """Split a pytest-benchmark name into a case name, rows and columns.

    Names look like `test_fit[n200-p10]` or
    `test_sweeps[n2000-p10-per_call]`: the parameters carrying a size are
    the size, and whatever else is there names the variant.
    """
    function, _, rest = name.partition("[")
    n = p = None
    variant = []
    for part in rest.rstrip("]").split("-"):
        match = SIZE.match(part)
        if match is None:
            variant.append(part)
        elif match.group(1) is not None:
            n = int(match.group(1))
        else:
            p = int(match.group(2))
    if n is None or p is None:
        raise SystemExit(f"no size in benchmark name {name!r}")
    case = "_".join([function.removeprefix("test_"), *variant])
    return case, n, p


def rows(core: dict, report: dict) -> list[dict]:
    out = []
    for benchmark in report["benchmarks"]:
        case, n, p = parse(benchmark["name"])
        reference = core_seconds(core, CORE_CASE[case.split("_")[0]], n, p)
        seconds = float(benchmark["stats"]["median"])
        out.append(
            {
                "case": case,
                "n": n,
                "p": p,
                "seconds": seconds,
                "core_seconds": reference,
                "overhead_seconds": seconds - reference,
                "ratio": seconds / reference,
            }
        )
    return sorted(out, key=lambda row: (row["n"], row["p"], row["case"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dir", type=Path, nargs="?", default=Path("target/bindings"))
    args = parser.parse_args()

    core = json.loads((args.dir / "core.json").read_text())
    report = json.loads((args.dir / "python.json").read_text())

    print("Python binding against the core, one machine, one session")
    print(
        f"core {core['core_version']}; sweeps {core['sweeps']}; "
        f"predict rows {core['predict_rows']}\n"
    )
    header = ("case", "n", "p", "seconds", "core_seconds", "overhead_seconds", "ratio")
    print(" ".join(f"{name:>18}" for name in header))
    for row in rows(core, report):
        print(
            f"{row['case']:>18} {row['n']:>18} {row['p']:>18} "
            f"{row['seconds']:>18.6g} {row['core_seconds']:>18.6g} "
            f"{row['overhead_seconds']:>18.6g} {row['ratio']:>18.3f}"
        )


if __name__ == "__main__":
    main()
