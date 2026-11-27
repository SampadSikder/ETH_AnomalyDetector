"""Repeat the frozen exploit benchmark across seeds, and report the spread.

A single-seed top-k table is one draw from a distribution. Three of the four
methods in this benchmark depend on a random seed - BlockScan's scorer masks a
random 15 % of each padded sequence, and the two trained models have seeded
initialisation and batch order - so a difference of one or two attacks in a
top-10 queue can be nothing but the seed. This module re-runs the benchmark
over a seed range and reports every run, the mean and the range, so a claim
about one model beating another has to survive the spread.

Rules this module holds to:

  * **every seed is kept.** There is no filtering, no best-of, and no dropping
    of a run that came out badly. A sweep that discards its unfavourable draws
    measures nothing;
  * **the seed-42 run is verified, not assumed.** Before the other seeds are
    launched, seed 42 is re-run and its top-k counts compared with the existing
    primary artifacts. A mismatch stops the sweep: if the configuration no
    longer reproduces the published numbers, the sweep would be measuring a
    different experiment;
  * **the original artifacts are never touched.** Each run writes to its own
    directory, and the primary `data/evaluation/detectors` tree is read for
    comparison only;
  * **the same hashes every time.** Each run is required to score exactly the
    frozen test hashes, and a run that does not is an error rather than a row
    with a smaller denominator;
  * **the effective seed is read back from inside the process** that used it,
    not copied from the request. A seed that silently failed to take effect
    would otherwise look like reproducibility.

Pairwise comparisons are reported as win / tie / loss counts over seeds rather
than as a difference of means, because with ten runs the mean's standard error
is wide and the per-seed sign is what a reader can check.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import EvaluationError
from .logs import get_logger
from . import topk as T

log = get_logger("seedsweep")

#: The methods swept. BlockScan is rescored; the other two are retrained.
SWEPT_METHODS = ("blockscan", "defiguard", "ft_transformer")

#: The seed of the primary run, whose artifacts are the reproduction reference.
PRIMARY_SEED = 42


@dataclass
class RunMetrics:
    """One (method, seed) run, scored on the frozen test hashes."""
    method: str
    seed: int
    effective_seed: int | None
    hashes: int
    average_precision: float | None
    at_k: dict[int, dict[str, Any]] = field(default_factory=dict)
    incidents_in_top10: int | None = None
    incident_names_in_top10: list[str] = field(default_factory=list)
    ties_crossing_cutoff: list[dict[str, Any]] = field(default_factory=list)
    note: str | None = None

    def attacks_at(self, k: int) -> int:
        return int(self.at_k[k]["attacks_found"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method, "seed": self.seed,
            "effective_seed": self.effective_seed,
            "hashes_ranked": self.hashes,
            "average_precision": self.average_precision,
            "attacks_at_k": {str(k): v["attacks_found"] for k, v in sorted(self.at_k.items())},
            "at_k": {str(k): v for k, v in sorted(self.at_k.items())},
            "distinct_attack_incidents_in_top10": self.incidents_in_top10,
            "attack_incidents_in_top10": self.incident_names_in_top10,
            "ties_crossing_cutoff": self.ties_crossing_cutoff,
            "note": self.note,
        }


def measure(scores: dict[str, float], labels: dict[str, int], *, method: str, seed: int,
            effective_seed: int | None, keys: Sequence[str], cutoffs: Iterable[int],
            incidents: dict[str, str], attack_classes: dict[str, str] | None = None,
            ) -> RunMetrics:
    """Rank one run and take every figure the sweep reports from it."""
    cutoffs = sorted(set(int(k) for k in cutoffs))
    ordered = T.rank(scores, labels, attack_classes=attack_classes, keys=keys)
    metrics = RunMetrics(method=method, seed=seed, effective_seed=effective_seed,
                         hashes=len(ordered),
                         average_precision=T.average_precision(ordered))
    for k in cutoffs:
        result = T.top_k(ordered, k, method=method)
        metrics.at_k[k] = {
            "attacks_found": result.attacks_found,
            "false_alerts": result.false_alerts,
            "precision_at_k": result.precision_at_k,
            "recall_at_k": result.recall_at_k,
            "false_positive_rate": result.false_positive_rate,
            "tie_at_cutoff": result.tie_at_cutoff,
            "attacks_found_min": result.attacks_found_min,
            "attacks_found_max": result.attacks_found_max,
            "selection_determined": result.selection_determined,
        }
        if result.tie_at_cutoff:
            metrics.ties_crossing_cutoff.append({
                "k": k, "cutoff_score": result.cutoff_score,
                "attacks_found": result.attacks_found,
                "reachable": [result.attacks_found_min, result.attacks_found_max],
                **{key: result.tied_block[key] for key in
                   ("tied_at_cutoff", "slots_remaining", "attacks_among_tied",
                    "negatives_among_tied", "expected_attacks_under_random_tie_order")},
            })
    # Distinct incidents among the attacks in the top ten: ten alerts tracing to
    # one incident is a narrower result than ten tracing to ten.
    if 10 in metrics.at_k:
        found = [r for r in ordered[:10] if r.label == 1]
        names = sorted({incidents.get(r.tx_key, r.tx_key) for r in found})
        metrics.incidents_in_top10 = len(names)
        metrics.incident_names_in_top10 = names
    return metrics


def reproduces(candidate: RunMetrics, reference: RunMetrics,
               *, cutoffs: Iterable[int]) -> dict[str, Any]:
    """Whether a re-run of the primary seed matches the existing artifacts."""
    checks = {}
    for k in sorted(set(int(c) for c in cutoffs)):
        got = candidate.at_k.get(k, {}).get("attacks_found")
        want = reference.at_k.get(k, {}).get("attacks_found")
        checks[f"attacks_at_{k}"] = {"primary": want, "rerun": got, "match": got == want}
    return {
        "method": candidate.method,
        "seed": candidate.seed,
        "checks": checks,
        "reproduced": all(c["match"] for c in checks.values()),
    }


def summarise(runs: Sequence[RunMetrics], *, k: int = 10) -> dict[str, Any]:
    """Per-method spread at one cutoff, over every seed that ran."""
    out: dict[str, Any] = {}
    for method in sorted({r.method for r in runs}):
        rows = sorted((r for r in runs if r.method == method), key=lambda r: r.seed)
        counts = [r.attacks_at(k) for r in rows]
        aps = [r.average_precision for r in rows if r.average_precision is not None]
        out[method] = {
            "seeds": [r.seed for r in rows],
            "attacks_at_k": counts,
            "k": k,
            "n_seeds": len(counts),
            "mean": statistics.fmean(counts) if counts else None,
            "median": statistics.median(counts) if counts else None,
            "min": min(counts) if counts else None,
            "max": max(counts) if counts else None,
            "range": (max(counts) - min(counts)) if counts else None,
            "stdev": statistics.stdev(counts) if len(counts) > 1 else 0.0,
            "average_precision_mean": statistics.fmean(aps) if aps else None,
            "average_precision_min": min(aps) if aps else None,
            "average_precision_max": max(aps) if aps else None,
            "incidents_in_top10": [r.incidents_in_top10 for r in rows],
        }
    return out


def head_to_head(runs: Sequence[RunMetrics], *, challenger: str, baseline: str,
                 k: int = 10) -> dict[str, Any]:
    """Per-seed sign test: how often `challenger` beats `baseline` at k."""
    by_seed: dict[int, dict[str, RunMetrics]] = {}
    for run in runs:
        by_seed.setdefault(run.seed, {})[run.method] = run
    paired = sorted(seed for seed, got in by_seed.items()
                    if challenger in got and baseline in got)
    if not paired:
        raise EvaluationError(f"no seed ran both {challenger} and {baseline}")
    wins = ties = losses = 0
    per_seed = []
    for seed in paired:
        a = by_seed[seed][challenger].attacks_at(k)
        b = by_seed[seed][baseline].attacks_at(k)
        wins += a > b
        ties += a == b
        losses += a < b
        per_seed.append({"seed": seed, challenger: a, baseline: b,
                         "margin": a - b,
                         "outcome": "win" if a > b else ("tie" if a == b else "loss")})
    margins = [row["margin"] for row in per_seed]
    return {
        "challenger": challenger, "baseline": baseline, "k": k,
        "seeds_compared": len(paired),
        "wins": wins, "ties": ties, "losses": losses,
        "win_rate": wins / len(paired),
        "mean_margin": statistics.fmean(margins),
        "margin_range": [min(margins), max(margins)],
        "per_seed": per_seed,
    }


def table(runs: Sequence[RunMetrics], cutoffs: Sequence[int]) -> str:
    """One row per (method, seed): the full sweep, including the bad draws."""
    header = ["method", "seed", "eff.seed"] + [f"@{k}" for k in cutoffs] + \
             ["AP", "incidents@10", "ties"]
    body = []
    for run in sorted(runs, key=lambda r: (r.method, r.seed)):
        body.append([run.method, str(run.seed),
                     "-" if run.effective_seed is None else str(run.effective_seed)]
                    + [str(run.attacks_at(k)) for k in cutoffs]
                    + ["-" if run.average_precision is None else f"{run.average_precision:.3f}",
                       "-" if run.incidents_in_top10 is None else str(run.incidents_in_top10),
                       ",".join(str(t["k"]) for t in run.ties_crossing_cutoff) or "-"])
    widths = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(header)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.ljust(w) for c, w in zip(b, widths)) for b in body]
    return "\n".join(lines)


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")
    return path
