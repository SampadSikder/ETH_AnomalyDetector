"""Top-k evaluation: what an analyst sees in a fixed-size alert queue.

This is the pooled form BlockScan reports in its Table 2. Every transaction in
a partition is ranked by one continuous score, the top *k* are taken, and the
attacks among them are counted. No threshold is chosen, which makes it the one
comparison that needs nothing from the validation partition - so a method with
no trained operating point, such as a raw length signal, competes on equal
terms with a fitted model.

Three rules keep the selection honest:

  * **ties are broken without labels.** Equal scores are ordered by ascending
    `tx_hash`, which is fixed by the data and cannot be nudged toward a better
    answer. Any label-aware tie-break would silently inflate precision@k;
  * **a tie that straddles the cutoff is reported as a range.** When the score
    at rank k repeats past rank k, the hash order decides which members get in,
    and that order is arbitrary with respect to the label. The realised count
    is reported with the minimum and maximum reachable by *any* valid ordering
    of that tied block, so a lucky or unlucky hash never reads as signal;
  * **`attacks + false alerts == k`,** asserted rather than assumed. The top-k
    population is the partition itself, so there is no third outcome.

Precision@k uses k as its denominator and recall@k the partition's attack
count, so with k far below the attack count recall is bounded well under 1 and
should be read as "share of attacks surfaced", not as a shortfall.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .errors import EvaluationError
from .logs import get_logger

log = get_logger("topk")


@dataclass(frozen=True)
class Ranked:
    """One transaction's place in one method's ranking."""
    rank: int
    tx_key: str
    tx_hash: str
    score: float
    label: int
    attack_class: str | None


@dataclass
class TopK:
    """One (method, k) cell of the table."""
    method: str
    k: int
    attacks_found: int
    false_alerts: int
    precision_at_k: float
    recall_at_k: float
    false_positive_rate: float
    attacks: int
    negatives: int
    tie_at_cutoff: bool
    attacks_found_min: int
    attacks_found_max: int
    cutoff_score: float
    selection_determined: bool = True
    expected_attacks_under_random_tie_order: float | None = None
    tied_block: dict[str, Any] | None = None
    selected: list[Ranked] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "method": self.method, "k": self.k,
            "attacks_found": self.attacks_found,
            "false_alerts": self.false_alerts,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "false_positive_rate": self.false_positive_rate,
            "attacks_in_partition": self.attacks,
            "negatives_in_partition": self.negatives,
            "cutoff_score": self.cutoff_score,
            "tie_at_cutoff": self.tie_at_cutoff,
            "attacks_found_min": self.attacks_found_min,
            "attacks_found_max": self.attacks_found_max,
            "selection_determined": self.selection_determined,
            "expected_attacks_under_random_tie_order":
                self.expected_attacks_under_random_tie_order,
            "tie_break": "ascending tx_hash; labels are not consulted",
        }
        if not self.selection_determined:
            out["warning"] = (
                "every selected transaction shares one score, so this row reports a "
                "hash-order lottery rather than a ranking: read the min-max range and "
                "the random-order expectation, not the realised count")
        if self.tied_block is not None:
            out["tied_block"] = self.tied_block
        return out


def rank(scores: dict[str, float], labels: dict[str, int],
         *, attack_classes: dict[str, str] | None = None,
         keys: Iterable[str] | None = None) -> list[Ranked]:
    """Order transactions by score, descending, ties by ascending tx_hash."""
    wanted = list(keys) if keys is not None else list(scores)
    missing = [k for k in wanted if k not in scores]
    if missing:
        raise EvaluationError(
            f"{len(missing)} of {len(wanted)} transactions have no score "
            f"(first: {missing[0]}): a top-k ranking needs every hash scored")
    unlabelled = [k for k in wanted if labels.get(k) is None]
    if unlabelled:
        raise EvaluationError(f"{len(unlabelled)} ranked transactions carry no label")
    classes = attack_classes or {}

    def tx_hash_of(key: str) -> str:
        return key.split(":", 1)[1] if ":" in key else key

    ordered = sorted(wanted, key=lambda k: (-float(scores[k]), tx_hash_of(k)))
    return [Ranked(rank=i + 1, tx_key=k, tx_hash=tx_hash_of(k),
                   score=float(scores[k]), label=int(labels[k]),
                   attack_class=classes.get(k))
            for i, k in enumerate(ordered)]


def _tie_bounds(ordered: Sequence[Ranked], k: int) -> tuple[bool, int, int, dict[str, Any] | None]:
    """Attack counts reachable at k over every valid ordering of the boundary tie."""
    cutoff = ordered[k - 1].score
    strictly_above = [r for r in ordered if r.score > cutoff]
    tied = [r for r in ordered if r.score == cutoff]
    slots = k - len(strictly_above)
    crosses = len(tied) > slots
    fixed_attacks = sum(r.label for r in strictly_above)
    tied_attacks = sum(r.label for r in tied)
    tied_negatives = len(tied) - tied_attacks
    if not crosses:
        return False, fixed_attacks + tied_attacks, fixed_attacks + tied_attacks, None
    # Drawing `slots` members from the tied block without replacement, the
    # expected number of attacks is slots * (attacks / block size). Reported
    # because a realised count that simply matches this expectation is evidence
    # the selection carries no signal at all.
    expected = slots * tied_attacks / len(tied)
    # Best case: fill every slot with a tied attack. Worst case: negatives first.
    lowest = fixed_attacks + max(0, slots - tied_negatives)
    highest = fixed_attacks + min(slots, tied_attacks)
    return True, lowest, highest, {
        "cutoff_score": cutoff,
        "ranks_fixed_above_cutoff": len(strictly_above),
        "tied_at_cutoff": len(tied),
        "slots_remaining": slots,
        "attacks_among_tied": tied_attacks,
        "negatives_among_tied": tied_negatives,
        "expected_attacks_under_random_tie_order": expected,
        "selection_determined": len(strictly_above) > 0,
        "note": ("the hash order decides which tied transactions are selected, and "
                 "that order is arbitrary with respect to the label"),
    }


def top_k(ordered: Sequence[Ranked], k: int, *, method: str) -> TopK:
    """One cell of the table, with the tie range when the cutoff splits a tie."""
    if k < 1:
        raise EvaluationError(f"k must be at least 1, got {k}")
    if k > len(ordered):
        raise EvaluationError(f"k={k} exceeds the {len(ordered)} ranked transactions")
    selected = list(ordered[:k])
    attacks_found = sum(r.label for r in selected)
    false_alerts = k - attacks_found
    attacks = sum(r.label for r in ordered)
    negatives = len(ordered) - attacks
    if attacks_found + false_alerts != k:
        raise EvaluationError(
            f"{method} at k={k}: {attacks_found} attacks + {false_alerts} false alerts "
            f"!= {k}; the top-k population must partition into exactly two outcomes")
    crosses, lowest, highest, block = _tie_bounds(ordered, k)
    if not (lowest <= attacks_found <= highest):
        raise EvaluationError(
            f"{method} at k={k}: realised {attacks_found} attacks lies outside the "
            f"reachable range [{lowest}, {highest}]")
    return TopK(
        method=method, k=k, attacks_found=attacks_found, false_alerts=false_alerts,
        precision_at_k=attacks_found / k,
        recall_at_k=(attacks_found / attacks) if attacks else 0.0,
        false_positive_rate=(false_alerts / negatives) if negatives else 0.0,
        attacks=attacks, negatives=negatives,
        tie_at_cutoff=crosses, attacks_found_min=lowest, attacks_found_max=highest,
        cutoff_score=ordered[k - 1].score,
        # A row whose every selected transaction shares one score is not a
        # ranking at all: the tie-break alone chose it. Saying so is the
        # difference between reporting a measurement and reporting a coin flip.
        selection_determined=bool(block is None or block["selection_determined"]),
        expected_attacks_under_random_tie_order=(
            None if block is None
            else sum(r.label for r in ordered if r.score > ordered[k - 1].score)
            + block["expected_attacks_under_random_tie_order"]),
        tied_block=block, selected=selected,
    )


def average_precision(ordered: Sequence[Ranked]) -> float | None:
    """AP over the same ranking, for cross-checking against the saved evaluation."""
    truth = [r.label for r in ordered]
    if len(set(truth)) < 2:
        return None
    try:
        from sklearn.metrics import average_precision_score
    except ImportError:                                 # pragma: no cover
        return None
    return float(average_precision_score(truth, [r.score for r in ordered]))


def overlap(a: Sequence[Ranked], b: Sequence[Ranked], *,
            b_full: Sequence[Ranked] | None = None) -> dict[str, Any]:
    """How much two methods' selections agree, and on what.

    When the second method's selection sits inside a tie, the realised overlap
    understates the agreement: the two can be picking from the same pool and
    still share nothing, because the tie-break alone decided which members of
    that pool `b` took. Passing `b_full` - `b`'s complete ranking - adds the
    overlap against the whole tied block, which is the comparison that survives
    the tie.
    """
    left = {r.tx_key for r in a}
    right = {r.tx_key for r in b}
    shared = left & right
    by_key = {r.tx_key: r for r in list(a) + list(b)}
    out_extra: dict[str, Any] = {}
    if b_full is not None and b:
        cutoff = min(r.score for r in b)
        block = [r for r in b_full if r.score == cutoff]
        if len(block) > len(b):
            in_block = [r for r in a if r.score is not None and r.tx_key in
                        {x.tx_key for x in block}]
            out_extra = {
                "second_method_tied_block": {
                    "score": cutoff, "size": len(block),
                    "attacks_in_block": sum(r.label for r in block),
                    "shared_with_block": len(in_block),
                    "shared_with_block_of": len(left),
                    "note": ("the second method cannot order inside this block, so "
                             "compare against the block rather than against the "
                             "particular members its tie-break happened to take"),
                }}
    return {**out_extra,
        "shared": len(shared),
        "shared_of": len(left),
        "jaccard": (len(shared) / len(left | right)) if (left | right) else None,
        "shared_attacks": sum(by_key[k].label for k in shared),
        "shared_hashes": sorted(by_key[k].tx_hash for k in shared),
        "only_left": sorted(by_key[k].tx_hash for k in left - right),
        "only_right": sorted(by_key[k].tx_hash for k in right - left),
    }


def table(rows: Sequence[TopK]) -> str:
    """The printed table. `attacks + false alerts == k` is visible in every row."""
    header = ["method", "k", "attacks", "false alerts", "P@k", "R@k", "FPR",
              "tie range", "E[tie]"]
    body = [[r.method + ("" if r.selection_determined else " *"),
             str(r.k), str(r.attacks_found), str(r.false_alerts),
             f"{r.precision_at_k:.3f}", f"{r.recall_at_k:.3f}",
             f"{r.false_positive_rate:.4f}",
             (f"{r.attacks_found_min}-{r.attacks_found_max}" if r.tie_at_cutoff else "-"),
             ("-" if r.expected_attacks_under_random_tie_order is None
              else f"{r.expected_attacks_under_random_tie_order:.1f}")]
            for r in rows]
    widths = [max(len(h), *(len(b[i]) for b in body)) for i, h in enumerate(header)]
    lines = ["  ".join(h.ljust(w) for h, w in zip(header, widths)),
             "  ".join("-" * w for w in widths)]
    lines += ["  ".join(c.ljust(w) for c, w in zip(b, widths)) for b in body]
    if any(not r.selection_determined for r in rows):
        lines += ["", "* every transaction in this row's selection shares one score: the "
                      "hash tie-break chose it, not the ranking."]
    return "\n".join(lines)


def selection_rows(result: TopK) -> list[dict[str, Any]]:
    """The selected hashes, with rank and score, for the saved artifact."""
    return [{"method": result.method, "k": result.k, "rank": r.rank,
             "tx_key": r.tx_key, "tx_hash": r.tx_hash, "score": r.score,
             "label": r.label, "attack_class": r.attack_class,
             "outcome": "attack" if r.label == 1 else "false_alert"}
            for r in result.selected]


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")
    return path
