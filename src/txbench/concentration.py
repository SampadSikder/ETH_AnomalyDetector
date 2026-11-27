"""Why a miss happened: incident concentration and shape-matched contrast.

A per-class recall of 0.000 is ambiguous on its own. It can mean the detectors
failed against many independent events, or that one event was missed over and
over because its transactions are near-duplicates of each other. The two
readings carry very different weight, so this module separates them.

Two measurements, both read-only:

  * **concentration** - how many distinct incidents produced a class's
    transactions, the largest incident's share, and the Herfindahl index of the
    per-incident shares. ``HHI == 1 / incidents`` means the transactions are
    spread evenly; ``HHI -> 1`` means one incident dominates. Incident-weighted
    recall is reported beside the per-transaction recall, because an incident is
    the unit a defender actually cares about.

  * **shape-matched contrast** - the class's score distribution against
    negatives *of the same transaction shape*, with every threshold left exactly
    as the evaluation froze it. Comparing a class of plain token transfers
    against a pool that is mostly multi-call DeFi traffic flatters a detector:
    it may be separating simple from complex rather than malicious from benign.
    Restricting the contrast to one shape removes that confound. An AUC near
    0.5 inside a single shape says the scores carry no information about the
    label at all, which is a stronger and more specific finding than a recall
    of zero.

Shape is read from the feature matrix, not re-derived: a transaction is a
``plain_erc20_transfer`` when the ERC-20 ``transfer`` selector fires and at most
``MAX_PLAIN_TOKENS`` token features are non-zero, and a ``bare_value_transfer``
when no selector and no log topic fires at all. Both are deliberately coarse -
they are a control group, not a taxonomy.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .errors import EvaluationError
from .logs import get_logger

log = get_logger("concentration")

#: ERC-20 ``transfer(address,uint256)``.
SEL_ERC20_TRANSFER = "sel:0xa9059cbb"

#: A plain transfer may fire a couple of extra tokens - the `Transfer` topic,
#: a `balanceOf` probe - and still be the same shape. Past this it is no longer
#: a fair control for a one-hop transfer.
MAX_PLAIN_TOKENS = 3

SHAPE_PLAIN_ERC20 = "plain_erc20_transfer"
SHAPE_BARE_VALUE = "bare_value_transfer"
SHAPE_OTHER = "other"


def _hhi(shares: Sequence[float]) -> float:
    return float(sum(s * s for s in shares))


@dataclass
class Concentration:
    """How few events produced a set of transactions."""
    attack_class: str
    transactions: int
    incidents: int
    largest_incident: str | None
    largest_incident_transactions: int
    largest_incident_share: float
    hhi: float
    even_hhi: float
    per_incident: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attack_class": self.attack_class,
            "transactions": self.transactions,
            "incidents": self.incidents,
            "largest_incident": self.largest_incident,
            "largest_incident_transactions": self.largest_incident_transactions,
            "largest_incident_share": self.largest_incident_share,
            "hhi": self.hhi,
            "hhi_if_even": self.even_hhi,
            "per_incident": self.per_incident,
        }


def concentration_of(manifest: pd.DataFrame, attack_class: str, *,
                     partition: str | None = None,
                     split_column: str = "split_temporal") -> Concentration:
    """Count the distinct incidents behind one attack class."""
    rows = manifest[manifest["attack_class"] == attack_class]
    if partition is not None:
        rows = rows[rows[split_column] == partition]
    if rows.empty:
        raise EvaluationError(
            f"no transactions of attack class {attack_class!r}"
            + (f" in partition {partition!r}" if partition else ""))
    counts = rows["group_incident"].astype(str).value_counts()
    total = int(counts.sum())
    shares = (counts / total).tolist()
    per_incident = []
    for name, count in counts.items():
        sub = rows[rows["group_incident"].astype(str) == name]
        stamps = sub["block_timestamp"].dropna().astype(str)
        per_incident.append({
            "incident": name,
            "protocol": (sub["group_protocol"].dropna().iloc[0]
                         if sub["group_protocol"].notna().any() else None),
            "transactions": int(count),
            "share": float(count) / total,
            "first_block": int(sub["block_number"].min()),
            "last_block": int(sub["block_number"].max()),
            "first_seen": stamps.min() if not stamps.empty else None,
            "last_seen": stamps.max() if not stamps.empty else None,
        })
    return Concentration(
        attack_class=attack_class,
        transactions=total,
        incidents=int(len(counts)),
        largest_incident=str(counts.index[0]),
        largest_incident_transactions=int(counts.iloc[0]),
        largest_incident_share=float(counts.iloc[0]) / total,
        hhi=_hhi(shares),
        even_hhi=1.0 / len(counts),
        per_incident=per_incident,
    )


def shapes_from_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Label each row's transaction shape from the feature matrix.

    The matrix is the model's own view of the transaction, so the shape is the
    one the model actually saw - not a second opinion derived from raw traces.
    """
    token_columns = [c for c in matrix.columns if ":" in c]
    if not token_columns:
        raise EvaluationError("feature matrix carries no token columns")
    non_zero = (matrix[token_columns] != 0).sum(axis=1)
    has_transfer = (matrix[SEL_ERC20_TRANSFER] != 0 if SEL_ERC20_TRANSFER in matrix.columns
                    else pd.Series(False, index=matrix.index))
    shape = pd.Series(SHAPE_OTHER, index=matrix.index, dtype=object)
    shape[has_transfer & (non_zero <= MAX_PLAIN_TOKENS)] = SHAPE_PLAIN_ERC20
    shape[non_zero == 0] = SHAPE_BARE_VALUE
    return pd.DataFrame({"tx_key": matrix["tx_key"], "tokens_fired": non_zero.astype(int),
                         "shape": shape.values})


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """P(a random positive outranks a random negative), ties counted as half.

    This is the Mann-Whitney statistic. It is reported instead of a mean
    difference because the scores are not on a common scale across models.
    """
    if not len(positive) or not len(negative):
        return None
    greater = sum(1 for p in positive for n in negative if p > n)
    equal = sum(1 for p in positive for n in negative if p == n)
    return (greater + 0.5 * equal) / (len(positive) * len(negative))


def _describe(scores: Sequence[float], threshold: float | None) -> dict[str, Any]:
    series = pd.Series(list(scores), dtype="float64")
    out: dict[str, Any] = {
        "n": int(len(series)),
        "min": float(series.min()), "p10": float(series.quantile(0.10)),
        "median": float(series.median()), "p90": float(series.quantile(0.90)),
        "max": float(series.max()), "mean": float(series.mean()),
        "distinct_values": int(series.round(9).nunique()),
    }
    if threshold is not None:
        out["flagged"] = int((series >= threshold).sum())
        out["flag_rate"] = float((series >= threshold).mean())
    return out


def contrast(scores: dict[str, float], labels: dict[str, int],
             attack_classes: dict[str, str], shapes: dict[str, str],
             *, attack_class: str, threshold: float | None,
             keys: Iterable[str],
             categories: dict[str, str] | None = None) -> dict[str, Any]:
    """One model's scores for a class, against negatives grouped and matched.

    ``threshold`` is applied exactly as handed in. Nothing here re-selects an
    operating point: the point of the contrast is to look underneath a frozen
    threshold, not to find a kinder one.
    """
    keys = [k for k in keys if k in scores]
    target = [k for k in keys if attack_classes.get(k) == attack_class]
    if not target:
        raise EvaluationError(f"no scored transactions of class {attack_class!r}")
    negatives = [k for k in keys if labels.get(k) == 0]
    target_shapes = pd.Series([shapes.get(k, SHAPE_OTHER) for k in target]).value_counts()
    dominant = str(target_shapes.index[0])

    def pick(ks: Iterable[str]) -> list[float]:
        return [scores[k] for k in ks]

    matched_negatives = [k for k in negatives if shapes.get(k) == dominant]
    result: dict[str, Any] = {
        "attack_class": attack_class,
        "threshold": threshold,
        "threshold_source": "frozen on the validation partition; applied unchanged",
        "class_scores": _describe(pick(target), threshold),
        "class_shapes": {str(k): int(v) for k, v in target_shapes.items()},
        "dominant_shape": dominant,
        "by_negative_group": {},
        "shape_matched": None,
    }
    for group in sorted({shapes.get(k, SHAPE_OTHER) for k in negatives}):
        members = [k for k in negatives if shapes.get(k, SHAPE_OTHER) == group]
        result["by_negative_group"][group] = {
            **_describe(pick(members), threshold),
            "auc_class_over_group": _auc(pick(target), pick(members)),
        }
    # The set's own negative taxonomy, reported beside the shape grouping: the
    # two answer different questions. `benign_category` says how a negative was
    # *sampled* (an ordinary transfer, a deliberately hard one), while the shape
    # says what the model saw. A class can look separable under one and not the
    # other, and both belong in the record.
    if categories:
        result["by_negative_category"] = {}
        for group in sorted({categories.get(k, "unknown") for k in negatives}):
            members = [k for k in negatives if categories.get(k, "unknown") == group]
            result["by_negative_category"][group] = {
                **_describe(pick(members), threshold),
                "auc_class_over_group": _auc(pick(target), pick(members)),
            }
    if matched_negatives:
        matched_target = [k for k in target if shapes.get(k) == dominant]
        result["shape_matched"] = {
            "shape": dominant,
            "class": _describe(pick(matched_target), threshold),
            "negatives": _describe(pick(matched_negatives), threshold),
            "auc": _auc(pick(matched_target), pick(matched_negatives)),
            "reading": ("an AUC near 0.5 means the scores do not separate this "
                        "class from benign transactions of the same shape"),
        }
    else:
        result["shape_matched_absent"] = (
            f"no labeled negative has the dominant shape {dominant!r}, so no "
            "shape-matched contrast is possible")
    return result


def incident_recall(scores: dict[str, float], attack_classes: dict[str, str],
                    incidents: dict[str, str], *, attack_class: str,
                    threshold: float | None) -> dict[str, Any]:
    """Detection counted per incident as well as per transaction."""
    keys = [k for k in scores if attack_classes.get(k) == attack_class]
    per: dict[str, bool] = {}
    for key in keys:
        incident = incidents.get(key, key)
        hit = threshold is not None and scores[key] >= threshold
        per[incident] = per.get(incident, False) or hit
    detected_tx = sum(1 for k in keys
                      if threshold is not None and scores[k] >= threshold)
    return {
        "transactions_scored": len(keys),
        "transactions_detected": detected_tx,
        "incidents": len(per),
        "incidents_detected": sum(1 for v in per.values() if v),
        "incident_recall": (sum(1 for v in per.values() if v) / len(per)) if per else None,
    }


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8")
    return path
