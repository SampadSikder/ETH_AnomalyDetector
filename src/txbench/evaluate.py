"""Scoring: strict prediction intake, honest metrics, and upstream parity.

Intake rules (a violation is refused, not warned about):
  * one row per eligible hash per (model, version, split): a score **or** an
    explicit failure status - silence is not a negative prediction;
  * duplicate rows for one key are refused;
  * a row that claims a test hash was in training is refused;
  * a score's direction must be declared, and is normalised so that higher
    always means "more likely an attack";
  * attack labels never enter a detector's input at scoring time: this module
    reads predictions and the frozen manifest, and the two meet only here.

Reporting rules:
  * the operating threshold is chosen on the **validation** partition only;
  * test results carry recall, precision, average precision, realised false
    positives per 1,000 labeled negatives, incident-weighted recall, and
    eligible / scored counts by class;
  * a binary-only detector is reported at its actual operating point;
  * a paired comparison runs on identical scored hashes, alongside each
    model's coverage of the full cohort.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .errors import EvaluationError
from .ids import normalize_chain_id, normalize_tx_hash, tx_key
from .logs import get_logger

log = get_logger("evaluate")

HIGHER_IS_ATTACK = "higher_is_attack"
LOWER_IS_ATTACK = "lower_is_attack"
DIRECTIONS = (HIGHER_IS_ATTACK, LOWER_IS_ATTACK)

STATUS_SCORED = "scored"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_UNSUPPORTED = "unsupported_input"
FAILURE_STATUSES = (STATUS_FAILED, STATUS_SKIPPED, STATUS_UNSUPPORTED)

REQUIRED_COLUMNS = ("chain_id", "tx_hash", "model", "version", "split", "status")
OPTIONAL_COLUMNS = ("score", "score_direction", "verdict", "note")


@dataclass
class Prediction:
    tx_key: str
    model: str
    version: str
    split: str
    status: str
    score: float | None = None
    direction: str = HIGHER_IS_ATTACK
    verdict: int | None = None
    note: str | None = None

    @property
    def oriented_score(self) -> float | None:
        if self.score is None:
            return None
        return self.score if self.direction == HIGHER_IS_ATTACK else -self.score


@dataclass
class PredictionSet:
    model: str
    version: str
    predictions: dict[str, Prediction] = field(default_factory=dict)
    direction: str = HIGHER_IS_ATTACK
    binary_only: bool = False
    problems: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.model}@{self.version}"


def read_predictions(path: Path, *, partitions: dict[str, str] | None = None,
                     eligible: Iterable[str] | None = None) -> dict[str, PredictionSet]:
    """Read and validate a predictions CSV. Raises on a refusable violation."""
    path = Path(path)
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise EvaluationError(f"{path} has no header row")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise EvaluationError(
                f"{path} is missing required column(s) {missing}. Required: "
                f"{list(REQUIRED_COLUMNS)}; optional: {list(OPTIONAL_COLUMNS)}")
        rows = list(reader)

    eligible_set = set(eligible) if eligible is not None else None
    sets: dict[str, PredictionSet] = {}
    seen: set[tuple[str, str, str]] = set()
    refusals: list[str] = []

    for i, row in enumerate(rows, start=2):
        try:
            chain_id = normalize_chain_id(row.get("chain_id"))
            tx_hash = normalize_tx_hash(row.get("tx_hash"))
        except Exception as exc:                        # noqa: BLE001 - reported, not raised yet
            refusals.append(f"row {i}: {exc}")
            continue
        key = tx_key(chain_id, tx_hash)
        model = (row.get("model") or "").strip()
        version = (row.get("version") or "").strip()
        if not model or not version:
            refusals.append(f"row {i}: model and version are required")
            continue
        identity = (model, version, key)
        if identity in seen:
            refusals.append(f"row {i}: duplicate prediction for {key} from {model}@{version}")
            continue
        seen.add(identity)

        declared_split = (row.get("split") or "").strip()
        if partitions is not None:
            actual = partitions.get(key)
            if actual is None:
                refusals.append(f"row {i}: {key} has no partition in the frozen split")
                continue
            if declared_split and declared_split != actual:
                if actual == "test" and declared_split in ("train", "training"):
                    refusals.append(
                        f"row {i}: {key} is a TEST hash but the prediction declares it "
                        f"'{declared_split}'. Refusing: test hashes must not appear in training.")
                else:
                    refusals.append(f"row {i}: {key} is '{actual}' in the frozen split but the "
                                    f"prediction declares '{declared_split}'")
                continue

        if eligible_set is not None and key not in eligible_set:
            refusals.append(f"row {i}: {key} is not an eligible hash for this task")
            continue

        status = (row.get("status") or "").strip().lower() or STATUS_SCORED
        direction = (row.get("score_direction") or "").strip().lower() or HIGHER_IS_ATTACK
        if direction not in DIRECTIONS:
            refusals.append(f"row {i}: score_direction must be one of {list(DIRECTIONS)}")
            continue

        score_text = (row.get("score") or "").strip()
        score = None
        if score_text:
            try:
                score = float(score_text)
            except ValueError:
                refusals.append(f"row {i}: score {score_text!r} is not a number")
                continue
        verdict_text = (row.get("verdict") or "").strip().lower()
        verdict = None
        if verdict_text:
            if verdict_text in ("1", "true", "yes", "attack"):
                verdict = 1
            elif verdict_text in ("0", "false", "no", "benign", "negative"):
                verdict = 0
            else:
                refusals.append(f"row {i}: verdict {verdict_text!r} is not binary")
                continue
        if status == STATUS_SCORED and score is None and verdict is None:
            refusals.append(f"row {i}: status 'scored' needs a score or a binary verdict")
            continue
        if status not in (STATUS_SCORED, *FAILURE_STATUSES):
            refusals.append(f"row {i}: status {status!r} is not one of "
                            f"{[STATUS_SCORED, *FAILURE_STATUSES]}")
            continue

        bundle = sets.setdefault(f"{model}@{version}", PredictionSet(model=model, version=version))
        bundle.direction = direction
        bundle.predictions[key] = Prediction(
            tx_key=key, model=model, version=version, split=declared_split or (
                partitions.get(key) if partitions else ""), status=status, score=score,
            direction=direction, verdict=verdict, note=(row.get("note") or "").strip() or None)

    if refusals:
        raise EvaluationError(f"{path}: refusing {len(refusals)} prediction row(s):\n  " +
                              "\n  ".join(refusals[:40]))
    for bundle in sets.values():
        scored = [p for p in bundle.predictions.values() if p.status == STATUS_SCORED]
        bundle.binary_only = bool(scored) and all(p.score is None for p in scored)
    return sets


def check_completeness(bundle: PredictionSet, eligible: Iterable[str]) -> dict[str, Any]:
    """One prediction or explicit failure per eligible hash - and nothing extra."""
    eligible_set = set(eligible)
    present = set(bundle.predictions)
    missing = sorted(eligible_set - present)
    extra = sorted(present - eligible_set)
    statuses = Counter(p.status for p in bundle.predictions.values())
    return {
        "model": bundle.label, "eligible": len(eligible_set), "rows": len(present),
        "missing_rows": len(missing), "missing_examples": missing[:20],
        "unexpected_rows": len(extra), "unexpected_examples": extra[:20],
        "status_counts": dict(statuses),
        "complete": not missing and not extra,
    }


# --- metrics ------------------------------------------------------------------

def _confusion(flags: dict[str, bool], labels: dict[str, int], keys: Iterable[str]
               ) -> dict[str, int]:
    tp = fp = fn = tn = 0
    for key in keys:
        label = labels.get(key)
        if label is None:
            continue
        hit = bool(flags.get(key))
        if label == 1:
            tp, fn = tp + hit, fn + (not hit)
        else:
            fp, tn = fp + hit, tn + (not hit)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _incident_weighted_recall(flags: dict[str, bool], labels: dict[str, int],
                              incidents: dict[str, str], keys: Iterable[str]) -> float | None:
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for key in keys:
        if labels.get(key) != 1:
            continue
        group = incidents.get(key, key)
        slot = groups[group]
        slot[0] += bool(flags.get(key))
        slot[1] += 1
    if not groups:
        return None
    return sum(hit / total for hit, total in groups.values()) / len(groups)


def select_threshold(bundle: PredictionSet, labels: dict[str, int], validation_keys: Iterable[str],
                     *, rule: str = "max_f1", target_fp_per_1k: float | None = None
                     ) -> dict[str, Any]:
    """Choose an operating threshold on validation rows only."""
    rows = [(p.oriented_score, labels.get(p.tx_key)) for key, p in bundle.predictions.items()
            if key in set(validation_keys) and p.status == STATUS_SCORED
            and p.oriented_score is not None and labels.get(key) is not None]
    if not rows:
        return {"rule": rule, "threshold": None,
                "reason": "no scored validation rows; a threshold cannot be chosen on "
                          "validation, so no thresholded test metric is reported"}
    candidates = sorted({score for score, _ in rows}, reverse=True)
    best: dict[str, Any] | None = None
    for threshold in candidates:
        tp = sum(1 for s, y in rows if s >= threshold and y == 1)
        fp = sum(1 for s, y in rows if s >= threshold and y == 0)
        fn = sum(1 for s, y in rows if s < threshold and y == 1)
        negatives = sum(1 for _, y in rows if y == 0)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        fp_per_1k = (fp / negatives * 1000) if negatives else None
        if rule == "target_fp_per_1k":
            if target_fp_per_1k is None:
                raise EvaluationError("rule 'target_fp_per_1k' needs target_fp_per_1k")
            if fp_per_1k is not None and fp_per_1k > target_fp_per_1k:
                continue
            score_key = (recall, -fp)
        else:
            score_key = (f1, recall)
        candidate = {"threshold": float(threshold), "validation_precision": precision,
                     "validation_recall": recall, "validation_f1": f1,
                     "validation_fp_per_1k_negatives": fp_per_1k}
        if best is None or score_key > best["_key"]:
            best = {**candidate, "_key": score_key}
    if best is None:
        return {"rule": rule, "threshold": None,
                "reason": f"no threshold satisfies the rule {rule!r} on validation"}
    best.pop("_key")
    return {"rule": rule, "fitted_on": "validation partition only",
            "validation_rows": len(rows), **best}


def _recall_by_attack_class(flags: dict[str, bool], labels: dict[str, int],
                            attack_classes: dict[str, str], keys: Iterable[str],
                            scored: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Recall per attack class, with the eligible and scored counts beside it.

    A class whose transactions were not all scored would otherwise look better
    or worse than it is, so the denominators are reported, not just the ratio.
    """
    scored = set(scored)
    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        if labels.get(key) != 1:
            continue
        name = attack_classes.get(key, "unclassified")
        slot = out.setdefault(name, {"eligible": 0, "scored": 0, "detected": 0})
        slot["eligible"] += 1
        if key in scored:
            slot["scored"] += 1
            slot["detected"] += int(bool(flags.get(key)))
    for slot in out.values():
        slot["recall"] = (slot["detected"] / slot["scored"]) if slot["scored"] else None
        slot["recall_over_eligible"] = (slot["detected"] / slot["eligible"]
                                        if slot["eligible"] else None)
    return out


def score_partition(bundle: PredictionSet, labels: dict[str, int], keys: Iterable[str],
                    *, threshold: float | None, incidents: dict[str, str] | None = None,
                    partition: str = "test",
                    attack_classes: dict[str, str] | None = None) -> dict[str, Any]:
    """Metrics for one partition at a fixed threshold (or a binary verdict)."""
    keys = list(keys)
    eligible_by_class = Counter(("attack" if labels.get(k) == 1 else
                                 "negative" if labels.get(k) == 0 else "unlabeled") for k in keys)
    scored_keys, flags = [], {}
    statuses = Counter()
    for key in keys:
        prediction = bundle.predictions.get(key)
        if prediction is None:
            statuses["absent"] += 1
            continue
        statuses[prediction.status] += 1
        if prediction.status != STATUS_SCORED:
            continue
        if prediction.verdict is not None and prediction.score is None:
            flags[key] = bool(prediction.verdict)
            scored_keys.append(key)
        elif prediction.oriented_score is not None and threshold is not None:
            flags[key] = prediction.oriented_score >= threshold
            scored_keys.append(key)
        elif prediction.verdict is not None:
            flags[key] = bool(prediction.verdict)
            scored_keys.append(key)

    confusion = _confusion(flags, labels, scored_keys)
    tp, fp, fn, tn = confusion["tp"], confusion["fp"], confusion["fn"], confusion["tn"]
    negatives = fp + tn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    out: dict[str, Any] = {
        "model": bundle.label, "partition": partition,
        "threshold": threshold, "binary_only_detector": bundle.binary_only,
        "eligible_by_class": dict(eligible_by_class),
        "scored_by_class": dict(Counter(("attack" if labels.get(k) == 1 else "negative")
                                        for k in scored_keys)),
        "prediction_status_counts": dict(statuses),
        **confusion,
        "precision": precision, "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)
               if precision and recall else None),
        # The raw count and its denominator, because "N false positives out of M
        # labeled negatives" is the number an operator actually feels.
        "false_positives": fp,
        "negatives_scored": negatives,
        "false_positives_per_1000_negatives": (fp / negatives * 1000) if negatives else None,
        "recall_by_attack_class": _recall_by_attack_class(
            flags, labels, attack_classes or {}, keys, scored_keys),
        "incident_weighted_recall": _incident_weighted_recall(
            flags, labels, incidents or {}, scored_keys),
        "coverage_of_partition": round(len(scored_keys) / len(keys), 6) if keys else None,
    }
    truth = [labels[k] for k in scored_keys if labels.get(k) is not None]
    scores = [bundle.predictions[k].oriented_score for k in scored_keys
              if labels.get(k) is not None]
    if truth and all(s is not None for s in scores) and len(set(truth)) > 1:
        try:
            from sklearn.metrics import average_precision_score
            out["average_precision"] = float(average_precision_score(truth, scores))
        except Exception as exc:                        # noqa: BLE001 - metric is optional
            out["average_precision"] = None
            out["average_precision_error"] = str(exc)
    else:
        out["average_precision"] = None
        out["average_precision_reason"] = (
            "a ranking metric needs continuous scores and both classes present among the "
            "scored rows" if not bundle.binary_only else
            "this detector emits a binary verdict only, so it is reported at its actual "
            "operating point and has no ranking metric")
    return out


def paired_comparison(results: dict[str, dict[str, Any]], bundles: dict[str, PredictionSet],
                      labels: dict[str, int], keys: Iterable[str], *,
                      thresholds: dict[str, float | None],
                      incidents: dict[str, str] | None = None,
                      attack_classes: dict[str, str] | None = None) -> dict[str, Any]:
    """Compare models on the hashes all of them actually scored."""
    keys = list(keys)
    scored_sets = {}
    for name, bundle in bundles.items():
        scored_sets[name] = {k for k in keys
                             if (p := bundle.predictions.get(k)) and p.status == STATUS_SCORED}
    if not scored_sets:
        return {"models": [], "shared_scored_hashes": 0}
    shared = set.intersection(*scored_sets.values()) if scored_sets else set()
    paired = {name: score_partition(bundles[name], labels, sorted(shared),
                                    threshold=thresholds.get(name), incidents=incidents,
                                    partition="test_shared", attack_classes=attack_classes)
              for name in bundles}
    return {
        "models": sorted(bundles),
        "partition_size": len(keys),
        "shared_scored_hashes": len(shared),
        "per_model_coverage_of_full_cohort": {
            name: {"scored": len(scored_sets[name]), "partition": len(keys),
                   "share": round(len(scored_sets[name]) / len(keys), 6) if keys else None}
            for name in sorted(scored_sets)},
        "paired_metrics": paired,
        "note": "the paired metrics use only hashes every model scored; the coverage block "
                "above shows what each model left unscored",
    }


# --- parity with the pinned upstream evaluator --------------------------------

def upstream_parity(evaluator_path: Path, benchmark_dir: Path, predictions_csv: Path, *,
                    split: str | None = None, track: str | None = None,
                    base_rate: float | None = None, tolerance: float = 1e-9,
                    python_executable: str | None = None) -> dict[str, Any]:
    """Run the release's own `evaluate.py` and compare it with a local replica.

    The local replica deliberately mirrors upstream's conventions, including
    the one this project otherwise refuses: upstream counts a benchmark row
    with no prediction as *not flagged*. Parity is checked against that rule;
    the project's own metrics keep requiring an explicit row per hash.
    """
    out: dict[str, Any] = {"evaluator": str(evaluator_path), "benchmark_dir": str(benchmark_dir),
                           "split": split, "track": track, "base_rate": base_rate,
                           "tolerance": tolerance}
    command = [python_executable or sys.executable, str(evaluator_path), str(predictions_csv),
               "--benchmark", str(benchmark_dir)]
    if split:
        command += ["--split", split]
    if track:
        command += ["--track", track]
    if base_rate is not None:
        command += ["--base-rate", str(base_rate)]
    out["command"] = command
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300,
                                   check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        out["status"] = "unverified"
        out["reason"] = f"could not run the upstream evaluator: {type(exc).__name__}: {exc}"
        return out
    if completed.returncode != 0:
        out["status"] = "unverified"
        out["reason"] = f"upstream evaluator exited {completed.returncode}"
        out["stderr"] = completed.stderr[-2000:]
        return out
    try:
        upstream = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        out["status"] = "unverified"
        out["reason"] = f"upstream output is not JSON: {exc}"
        out["stdout"] = completed.stdout[-2000:]
        return out

    local = local_replica(benchmark_dir, predictions_csv, split=split, track=track,
                          base_rate=base_rate)
    shared = sorted(set(upstream) & set(local))
    differences: dict[str, Any] = {}
    for key in shared:
        a, b = upstream[key], local[key]
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and \
                not isinstance(a, bool) and not isinstance(b, bool):
            if abs(float(a) - float(b)) > tolerance:
                differences[key] = {"upstream": a, "local": b,
                                    "abs_difference": abs(float(a) - float(b))}
        elif a != b:
            differences[key] = {"upstream": a, "local": b}
    out.update({
        "status": "verified" if not differences else "mismatch",
        "upstream_result": upstream,
        "local_result": local,
        "compared_keys": shared,
        "keys_only_upstream": sorted(set(upstream) - set(local)),
        "keys_only_local": sorted(set(local) - set(upstream)),
        "differences": differences,
        "convention_checked": "a benchmark row without a prediction counts as not flagged "
                              "(upstream's rule), incidents weighted equally, track membership "
                              "keeps every negative and filters attacks by category",
    })
    return out


def local_replica(benchmark_dir: Path, predictions_csv: Path, *, split: str | None,
                  track: str | None, base_rate: float | None) -> dict[str, Any]:
    """A local re-implementation of upstream's scoring, for the parity check."""
    with open(Path(benchmark_dir) / "combined.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r.get(f"split_{split}") == "test"]
    if track:
        rows = [r for r in rows if r["label"] == "0" or r["category"] == track]
    with open(predictions_csv, encoding="utf-8-sig", newline="") as f:
        flagged = {}
        for row in csv.DictReader(f):
            key = (row.get("tx_hash") or "").strip().lower()
            raw = (row.get("prediction") or row.get("verdict") or "").strip().lower()
            if not raw and (row.get("status") or "").strip().lower() in FAILURE_STATUSES:
                raw = "0"
            flagged[key] = raw in ("1", "true", "yes", "attack")

    tp = fp = fn = tn = 0
    incidents: dict[str, list[int]] = {}
    for row in rows:
        hit = flagged.get(row["tx_hash"].lower(), False)
        if row["label"] == "1":
            tp, fn = tp + hit, fn + (not hit)
            slot = incidents.setdefault(row["group_incident"], [0, 0])
            slot[0] += hit
            slot[1] += 1
        else:
            fp, tn = fp + hit, tn + (not hit)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    out = {"split": split or "all rows", "track": track or "all attacks",
           "rows": len(rows), "attacks": tp + fn, "negatives": fp + tn,
           "incidents": len(incidents),
           "without_prediction": sum(r["tx_hash"].lower() not in flagged for r in rows),
           "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall,
           "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
           "false_positive_rate": fpr,
           "recall_incidents_equal": (sum(h / n for h, n in incidents.values()) / len(incidents)
                                      if incidents else 0.0)}
    if base_rate is not None:
        flagged_share = recall * base_rate + fpr * (1 - base_rate)
        out["base_rate"] = base_rate
        out["precision_at_base_rate"] = (recall * base_rate / flagged_share
                                         if flagged_share else 0.0)
    return out


def write_upstream_benchmark_subset(combined_csv: Path, out_dir: Path, *,
                                    chain_id: int = 1) -> Path:
    """An Ethereum-only `combined.csv` so the pinned evaluator scores our cohort.

    The upstream file is never modified; a filtered copy is written, with the
    same columns, into a working directory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(combined_csv, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [r for r in reader if str(r.get("chain_id", "")).strip() == str(chain_id)]
    target = out_dir / "combined.csv"
    tmp = target.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(target)
    log.info("wrote %d chain-%d rows to %s for the upstream evaluator",
             len(rows), chain_id, target)
    return target


def to_upstream_predictions(bundle: PredictionSet, keys: Iterable[str], threshold: float | None,
                            out_path: Path) -> Path:
    """Write a `tx_hash,prediction` file in the upstream evaluator's format."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tx_hash", "prediction"])
        for key in keys:
            prediction = bundle.predictions.get(key)
            if prediction is None or prediction.status != STATUS_SCORED:
                continue
            if prediction.verdict is not None and prediction.score is None:
                flag = int(bool(prediction.verdict))
            elif prediction.oriented_score is not None and threshold is not None:
                flag = int(prediction.oriented_score >= threshold)
            elif prediction.verdict is not None:
                flag = int(bool(prediction.verdict))
            else:
                continue
            writer.writerow([key.split(":", 1)[1], flag])
    tmp.replace(out_path)
    return out_path


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
