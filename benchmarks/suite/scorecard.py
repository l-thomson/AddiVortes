"""The scorecard of one suite cell.

Every diagnostic comes from ArviZ, pinned in `requirements.txt`:
rank-normalised split-chain R-hat, bulk effective sample size and tail
effective sample size as defined by Vehtari, Gelman, Simpson, Carpenter
and Buerkner (2021). No estimator is written here. ESS estimators disagree
materially on poorly mixed chains, so which one produced a number is part
of the number.

The currency is minimum ESS per second over the inferential quantities,
bulk and tail. Wall-clock alone is the wrong currency for a Markov chain: a
sampler twice as fast per sweep that mixes half as well has gained
nothing. ESS per sweep is reported beside it, so algorithmic efficiency
stays separable from implementation speed and the pair survives a change
of hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import arviz as az
import numpy as np
import pandas as pd

#: The effective sample size a practitioner is taken to want, for the
#: time-to-target metric. Vehtari et al (2021) call 400 the point below
#: which the ESS estimate itself is unreliable.
TARGET_ESS = 400.0

#: The reporting threshold for rank-normalised R-hat (Vehtari et al 2021,
#: s. 4). Recorded in the note beside `rhat_max`; nothing gates on it,
#: because a cell whose R-hat sits near it flips from run to run and a
#: gate that flips is noise.
RHAT_REPORT = 1.01

#: R-hat above this and the chains have not converged at all, so the
#: efficiency numbers of that cell describe nothing. The gate reads this
#: and the change in R-hat against the baseline, not the reporting
#: threshold.
RHAT_LIMIT = 1.05

#: The structure counts: mean cells and mean active dimensions per
#: tessellation. They are reported, and they do not enter the currency or
#: the validity gate. A sum of tessellations is not identified, so the
#: structure of any one draw wanders while the function it encodes does
#: not; the BART family monitors sigma and f(x) for the same reason.
STRUCTURE = ("cells", "dims")


@dataclass(frozen=True)
class Run:
    """One repetition of one cell: its chains, its draws and its timings."""

    cell: str
    model: str
    n: int
    p: int
    seed: int
    draws: pd.DataFrame
    metadata: list[dict]
    peak_rss_bytes: int

    @property
    def chains(self) -> int:
        return len(self.metadata)

    @property
    def n_draws(self) -> int:
        return int(self.metadata[0]["draws"])

    @property
    def post_warmup_seconds(self) -> float:
        """Post-warm-up sampling time over all chains, run one at a time."""
        return sum(m["post_warmup_seconds"] for m in self.metadata)

    @property
    def warmup_seconds(self) -> float:
        return sum(m["warmup_seconds"] for m in self.metadata)

    @property
    def fit_seconds(self) -> float:
        return sum(m["fit_seconds"] for m in self.metadata)

    @property
    def predict_seconds(self) -> float:
        return float(np.mean([m["predict_seconds"] for m in self.metadata]))

    @property
    def sweeps(self) -> int:
        """Post-warm-up sweeps over all chains."""
        return self.n_draws * self.chains


def posterior(draws: pd.DataFrame) -> Any:
    """Return the tidy draws as an ArviZ posterior group.

    Parameters
    ----------
    draws : pandas.DataFrame
        Columns `chain`, `draw`, `quantity`, `value`, as the suite binary
        writes them.
    """
    wide = draws.pivot_table(
        index=["chain", "draw"], columns="quantity", values="value"
    ).sort_index()
    chains = wide.index.get_level_values("chain").unique().size
    n_draws = wide.index.get_level_values("draw").unique().size
    group = {
        str(name): wide[name].to_numpy().reshape(chains, n_draws)
        for name in wide.columns
    }
    return az.from_dict({"posterior": group})


def _per_quantity(diagnostic: Any) -> dict[str, float]:
    """Flatten an ArviZ diagnostic to one value per quantity."""
    return {
        str(name): float(np.ravel(np.asarray(variable.values))[0])
        for name, variable in diagnostic.dataset.data_vars.items()
    }


def _declared(flat: dict[str, float]) -> dict[str, float]:
    """The inferential quantities: sigma and f(x), never the structure."""
    return {name: value for name, value in flat.items() if name not in STRUCTURE}


def _minimum(flat: dict[str, float]) -> tuple[float, str]:
    """The smallest value in `flat` and the quantity carrying it."""
    quantity = min(flat, key=lambda k: flat[k])
    return flat[quantity], quantity


def scorecard(run: Run) -> pd.DataFrame:
    """Return the scorecard of `run`, one row per metric."""
    tree = posterior(run.draws)
    rhat = _per_quantity(az.rhat(tree, method="rank"))
    ess_bulk = _per_quantity(az.ess(tree, method="bulk"))
    ess_tail = _per_quantity(az.ess(tree, method="tail"))

    rhat_max = max(_declared(rhat).values())
    bulk, bulk_quantity = _minimum(_declared(ess_bulk))
    tail, tail_quantity = _minimum(_declared(ess_tail))

    seconds = run.post_warmup_seconds
    rows: list[dict] = []

    def add(metric: str, value: float | None, unit: str, note: str = "") -> None:
        rows.append({"metric": metric, "value": value, "unit": unit, "note": note})

    add("rhat_max", rhat_max, "ratio", f"reporting threshold {RHAT_REPORT}")
    add("ess_bulk_min", bulk, "draws", bulk_quantity)
    add("ess_tail_min", tail, "draws", tail_quantity)
    add("ess_bulk_min_per_second", bulk / seconds, "1/s", bulk_quantity)
    add("ess_tail_min_per_second", tail / seconds, "1/s", tail_quantity)
    add("ess_bulk_min_per_sweep", bulk / run.sweeps, "ratio", bulk_quantity)
    add("ess_tail_min_per_sweep", tail / run.sweeps, "ratio", tail_quantity)
    add(
        "seconds_per_sweep",
        seconds / run.sweeps,
        "s",
        "post-warm-up, initialisation excluded",
    )
    add(
        "seconds_to_target_ess",
        run.warmup_seconds / run.chains + seconds * TARGET_ESS / bulk,
        "s",
        f"bulk ESS {TARGET_ESS:.0f}, one chain at a time",
    )
    add("fit_seconds", run.fit_seconds / run.chains, "s", "per chain")
    add("predict_seconds", run.predict_seconds, "s", "held-out design")
    add("peak_rss_mb", run.peak_rss_bytes / 1e6, "MB", "one chain process")

    # The Monte Carlo standard error of the posterior mean of each
    # inferential quantity, the largest reported: a scorecard without it
    # invites a comparison of two numbers that differ by less than their
    # own noise.
    add(
        "mcse_mean_max",
        max(_declared(_per_quantity(az.mcse(tree, method="mean"))).values()),
        "quantity units",
    )

    structure_rhat = {name: rhat[name] for name in STRUCTURE if name in rhat}
    structure_ess = {name: ess_bulk[name] for name in STRUCTURE if name in ess_bulk}
    if structure_rhat:
        add("rhat_max_structure", max(structure_rhat.values()), "ratio", "reported")
    if structure_ess:
        add("ess_bulk_min_structure", min(structure_ess.values()), "draws", "reported")

    accuracy = [m["accuracy"] for m in run.metadata if m.get("accuracy")]
    if accuracy:
        for key, unit in (
            ("rmse", "response units"),
            ("lpd", "log density"),
            ("coverage_95", "share"),
            ("width_95", "response units"),
        ):
            values = [a[key] for a in accuracy if a.get(key) is not None]
            if values:
                add(key, float(np.mean(values)), unit, "held-out")

    frame = pd.DataFrame(rows)
    frame.insert(0, "cell", run.cell)
    frame.insert(1, "model", run.model)
    frame.insert(2, "n", run.n)
    frame.insert(3, "p", run.p)
    frame.insert(4, "seed", run.seed)
    return frame


def summarise(scorecards: list[pd.DataFrame]) -> pd.DataFrame:
    """Return the mean and standard error of each metric over repetitions.

    Parameters
    ----------
    scorecards : list of pandas.DataFrame
        One scorecard per repetition of the same cell.
    """
    joined = pd.concat(scorecards, ignore_index=True)
    grouped = joined.groupby(
        ["cell", "model", "n", "p", "metric", "unit"], as_index=False
    )
    out = grouped.agg(
        value=("value", "mean"),
        se=(
            "value",
            lambda v: (
                float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0
            ),
        ),
        reps=("value", "size"),
    )
    notes = joined.drop_duplicates(subset=["cell", "metric"])[
        ["cell", "metric", "note"]
    ]
    return out.merge(notes, on=["cell", "metric"], how="left")


def relative_standard_error(scorecards: list[pd.DataFrame], metric: str) -> float:
    """Return the standard error of `metric` over repetitions, over its mean.

    Used to size the number of repetitions from the variance observed so
    far rather than from a round number chosen in advance.
    """
    values = [
        float(card.loc[card["metric"] == metric, "value"].iloc[0])
        for card in scorecards
    ]
    if len(values) < 2:
        return math.inf
    mean = float(np.mean(values))
    if mean == 0.0:
        return math.inf
    return float(np.std(values, ddof=1) / math.sqrt(len(values)) / abs(mean))
