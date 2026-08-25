#!/usr/bin/env python3
"""Render the instruction-count comparison as a markdown table.

Usage: tools/perf-instructions-table.py <gungraun-dir>

Reads every summary.json that `cargo bench --bench instructions --
--baseline=base --save-summary` wrote under the gungraun output directory
(one per benchmark) and prints one row per benchmark: the base revision's
count, the pull request's count and the change, for retired instructions
and estimated cycles. A benchmark with no base value (a benchmark the pull
request adds) shows a dash in the base and change columns. A row whose
soft limit was breached says so in its change column.

Exits 1 when no summary is found, since a run that wrote nothing is a
broken run rather than an empty comparison.
"""

import json
import re
import sys
from pathlib import Path

METRICS = [("Ir", "instructions"), ("EstimatedCycles", "est. cycles")]


def metric_value(metric):
    """The number inside a gungraun `Metric` (`{"Int": n}` or `{"Float": x}`)."""
    if "Int" in metric:
        return metric["Int"]
    return metric["Float"]


def sides(diff):
    """(new, old) of a `MetricsDiff`, either side None when absent."""
    metrics = diff["metrics"]
    if "Both" in metrics:
        new, old = metrics["Both"]
        return metric_value(new), metric_value(old)
    if "Left" in metrics:
        return metric_value(metrics["Left"]), None
    return None, metric_value(metrics["Right"])


def bench_name(summary):
    """`sweep/gaussian` from function `sweep` and details `sampler("gaussian")`.

    Falls back to the function name and the benchmark id when the details
    do not carry a single argument.
    """
    function = summary["function_name"]
    details = summary.get("details") or ""
    match = re.search(r"\(([^()]*)\)", details)
    if match:
        argument = match.group(1).strip().strip("\"'")
        if argument:
            return f"{function}/{argument}"
    if summary.get("id"):
        return f"{function}/{summary['id']}"
    return function


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.1f}"
    return f"{value:,}"


def change(new, old, breached):
    """`20.7% fewer`, `6.8% more` or `unchanged`: a word rather than a
    sign carries the direction."""
    if new is None or old is None:
        return "-"
    if old == 0:
        text = "n/a"
    elif new == old:
        text = "unchanged"
    else:
        pct = 100.0 * (new - old) / old
        text = f"{abs(pct):.1f}% {'fewer' if pct < 0 else 'more'}"
    return f"{text} (limit breached)" if breached else text


def row(summary):
    callgrind = None
    breached = set()
    for profile in summary["profiles"]:
        total = profile["summaries"]["total"]
        tool_summary = total["summary"]
        if isinstance(tool_summary, dict) and "Callgrind" in tool_summary:
            callgrind = tool_summary["Callgrind"]
            for regression in total["regressions"]:
                kind = regression.get("Soft") or regression.get("Hard") or {}
                metric = kind.get("metric")
                if isinstance(metric, dict):
                    metric = next(iter(metric.values()))
                breached.add(metric)
            break
    if callgrind is None:
        return None
    cells = [bench_name(summary)]
    for key, _ in METRICS:
        diff = callgrind.get(key)
        if diff is None:
            cells += ["-", "-", "-"]
            continue
        new, old = sides(diff)
        cells += [fmt(old), fmt(new), change(new, old, key in breached)]
    return cells


def main(argv):
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(argv[1])
    summaries = []
    for path in sorted(root.rglob("summary.json")):
        with open(path, encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    rows = [r for r in (row(s) for s in summaries) if r is not None]
    if not rows:
        print(f"no callgrind summary under {root}", file=sys.stderr)
        return 1
    baselines = {s["baselines"][1] for s in summaries if s["baselines"][1]}
    base = ", ".join(sorted(baselines)) or "previous run"
    print("### Instruction counts")
    print()
    print(
        f"Callgrind, one run per benchmark, base revision (`{base}`) "
        "measured with this pull request's benchmark code. "
        "Fewer is less work; the gate fails at more than 5% more in "
        "either column."
    )
    print()
    header = ["bench"]
    for _, label in METRICS:
        header += [f"{label} base", f"{label} PR", "change"]
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join([" -- "] + [" --: "] * (len(header) - 1)) + "|")
    for cells in sorted(rows, key=lambda r: r[0]):
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
