"""Prediction intake and metrics: refusals, validation-only thresholds, parity."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from txbench import evaluate as E
from txbench.errors import EvaluationError

KEYS = [f"1:0x{i:064x}" for i in range(1, 9)]
LABELS = {KEYS[0]: 1, KEYS[1]: 0, KEYS[2]: 1, KEYS[3]: 0,
          KEYS[4]: 1, KEYS[5]: 0, KEYS[6]: 1, KEYS[7]: 0}
PARTITIONS = {KEYS[0]: "train", KEYS[1]: "train",
              KEYS[2]: "validation", KEYS[3]: "validation",
              KEYS[4]: "test", KEYS[5]: "test", KEYS[6]: "test", KEYS[7]: "test"}
INCIDENTS = {KEYS[0]: "i1", KEYS[2]: "i2", KEYS[4]: "i3", KEYS[6]: "i3"}

HEADER = ["chain_id", "tx_hash", "model", "version", "split", "score", "score_direction",
          "verdict", "status", "note"]


def write_predictions(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: "" for k in HEADER}, **row})
    return path


def row(key, score=None, *, model="m", version="1", status="scored", verdict="",
        direction="higher_is_attack", split=None):
    chain, tx_hash = key.split(":", 1)
    return {"chain_id": chain, "tx_hash": tx_hash, "model": model, "version": version,
            "split": split if split is not None else PARTITIONS[key],
            "score": "" if score is None else str(score), "score_direction": direction,
            "verdict": verdict, "status": status}


def full_set(tmp_path, **kwargs):
    rows = [row(k, score=0.9 if LABELS[k] == 1 else 0.1, **kwargs) for k in KEYS]
    return write_predictions(tmp_path / "p.csv", rows)


def test_duplicate_prediction_is_refused(tmp_path):
    path = write_predictions(tmp_path / "p.csv", [row(KEYS[0], 0.5), row(KEYS[0], 0.6)])
    with pytest.raises(EvaluationError, match="duplicate prediction"):
        E.read_predictions(path, partitions=PARTITIONS)


def test_a_test_hash_declared_as_training_is_refused(tmp_path):
    path = write_predictions(tmp_path / "p.csv", [row(KEYS[4], 0.5, split="train")])
    with pytest.raises(EvaluationError, match="test hashes must not appear in training"):
        E.read_predictions(path, partitions=PARTITIONS)


def test_a_scored_row_needs_a_score_or_a_verdict(tmp_path):
    path = write_predictions(tmp_path / "p.csv", [row(KEYS[0])])
    with pytest.raises(EvaluationError, match="needs a score or a binary verdict"):
        E.read_predictions(path, partitions=PARTITIONS)


def test_an_unknown_score_direction_is_refused(tmp_path):
    path = write_predictions(tmp_path / "p.csv", [row(KEYS[0], 0.5, direction="bigger_better")])
    with pytest.raises(EvaluationError, match="score_direction"):
        E.read_predictions(path, partitions=PARTITIONS)


def test_a_malformed_hash_is_refused(tmp_path):
    path = write_predictions(tmp_path / "p.csv",
                             [{"chain_id": "1", "tx_hash": "0xnothex", "model": "m",
                               "version": "1", "split": "test", "score": "0.5",
                               "status": "scored"}])
    with pytest.raises(EvaluationError):
        E.read_predictions(path, partitions=PARTITIONS)


def test_missing_rows_are_reported_rather_than_counted_as_negative(tmp_path):
    path = write_predictions(tmp_path / "p.csv", [row(KEYS[4], 0.9)])
    bundles = E.read_predictions(path, partitions=PARTITIONS)
    completeness = E.check_completeness(bundles["m@1"], KEYS)
    assert completeness["complete"] is False
    assert completeness["missing_rows"] == len(KEYS) - 1
    result = E.score_partition(bundles["m@1"], LABELS, [KEYS[4], KEYS[5]], threshold=0.5)
    assert result["prediction_status_counts"]["absent"] == 1
    assert result["tp"] + result["fp"] + result["fn"] + result["tn"] == 1, \
        "an absent row is not scored at all, rather than scored as a negative"


def test_an_explicit_failure_is_accepted_and_kept_out_of_the_metrics(tmp_path):
    rows = [row(k, score=0.9 if LABELS[k] == 1 else 0.1) for k in KEYS]
    rows[4] = row(KEYS[4], status="failed")
    path = write_predictions(tmp_path / "p.csv", rows)
    bundles = E.read_predictions(path, partitions=PARTITIONS)
    assert E.check_completeness(bundles["m@1"], KEYS)["complete"] is True
    result = E.score_partition(bundles["m@1"], LABELS, [k for k in KEYS
                                                        if PARTITIONS[k] == "test"],
                               threshold=0.5)
    assert result["prediction_status_counts"]["failed"] == 1
    assert result["scored_by_class"]["attack"] == 1, "the failed attack row is not scored"


def test_lower_is_attack_scores_are_oriented(tmp_path):
    rows = [row(k, score=0.1 if LABELS[k] == 1 else 0.9, direction="lower_is_attack")
            for k in KEYS]
    path = write_predictions(tmp_path / "p.csv", rows)
    bundles = E.read_predictions(path, partitions=PARTITIONS)
    chosen = E.select_threshold(bundles["m@1"], LABELS,
                                [k for k in KEYS if PARTITIONS[k] == "validation"])
    result = E.score_partition(bundles["m@1"], LABELS,
                               [k for k in KEYS if PARTITIONS[k] == "test"],
                               threshold=chosen["threshold"])
    assert result["recall"] == 1.0 and result["precision"] == 1.0


def test_threshold_is_chosen_on_validation_only(tmp_path):
    path = full_set(tmp_path)
    bundles = E.read_predictions(path, partitions=PARTITIONS)
    validation = [k for k in KEYS if PARTITIONS[k] == "validation"]
    chosen = E.select_threshold(bundles["m@1"], LABELS, validation)
    assert chosen["fitted_on"] == "validation partition only"
    assert chosen["validation_rows"] == len(validation)
    assert chosen["threshold"] is not None


def test_no_scored_validation_rows_means_no_threshold(tmp_path):
    rows = [row(k, score=0.5) if PARTITIONS[k] != "validation" else row(k, status="failed")
            for k in KEYS]
    path = write_predictions(tmp_path / "p.csv", rows)
    bundles = E.read_predictions(path, partitions=PARTITIONS)
    chosen = E.select_threshold(bundles["m@1"], LABELS,
                                [k for k in KEYS if PARTITIONS[k] == "validation"])
    assert chosen["threshold"] is None
    assert "cannot be chosen on validation" in chosen["reason"]


def test_test_metrics_include_the_required_quantities(tmp_path):
    path = full_set(tmp_path)
    bundles = E.read_predictions(path, partitions=PARTITIONS)
    test = [k for k in KEYS if PARTITIONS[k] == "test"]
    result = E.score_partition(bundles["m@1"], LABELS, test, threshold=0.5, incidents=INCIDENTS)
    for field in ("recall", "precision", "average_precision",
                  "false_positives_per_1000_negatives", "incident_weighted_recall",
                  "eligible_by_class", "scored_by_class"):
        assert field in result
    assert result["recall"] == 1.0
    assert result["false_positives_per_1000_negatives"] == 0.0
    # two attacks share incident i3, so incident-weighted recall averages over 1 incident
    assert result["incident_weighted_recall"] == 1.0


def test_false_positives_per_1000_negatives_is_realised_not_modelled(tmp_path):
    rows = [row(k, score=0.9) for k in KEYS]          # flags everything
    path = write_predictions(tmp_path / "p.csv", rows)
    bundles = E.read_predictions(path, partitions=PARTITIONS)
    test = [k for k in KEYS if PARTITIONS[k] == "test"]
    result = E.score_partition(bundles["m@1"], LABELS, test, threshold=0.5)
    assert result["fp"] == 2 and result["tn"] == 0
    assert result["false_positives_per_1000_negatives"] == 1000.0


def test_binary_only_detector_is_reported_at_its_operating_point(tmp_path):
    rows = [row(k, verdict="1" if LABELS[k] == 1 else "0") for k in KEYS]
    path = write_predictions(tmp_path / "p.csv", rows)
    bundles = E.read_predictions(path, partitions=PARTITIONS)
    bundle = bundles["m@1"]
    assert bundle.binary_only is True
    result = E.score_partition(bundle, LABELS, [k for k in KEYS if PARTITIONS[k] == "test"],
                               threshold=None)
    assert result["binary_only_detector"] is True
    assert result["recall"] == 1.0
    assert result["average_precision"] is None
    assert "binary verdict only" in result["average_precision_reason"]


def test_paired_comparison_uses_only_shared_scored_hashes(tmp_path):
    rows = [row(k, score=0.9 if LABELS[k] == 1 else 0.1, model="a") for k in KEYS]
    rows += [row(k, score=0.8 if LABELS[k] == 1 else 0.2, model="b") for k in KEYS[:6]]
    rows += [row(k, status="failed", model="b") for k in KEYS[6:]]
    path = write_predictions(tmp_path / "p.csv", rows)
    bundles = E.read_predictions(path, partitions=PARTITIONS)
    test = [k for k in KEYS if PARTITIONS[k] == "test"]
    thresholds = {name: 0.5 for name in bundles}
    paired = E.paired_comparison({}, bundles, LABELS, test, thresholds=thresholds)
    assert paired["shared_scored_hashes"] == 2
    assert paired["per_model_coverage_of_full_cohort"]["a@1"]["scored"] == 4
    assert paired["per_model_coverage_of_full_cohort"]["b@1"]["scored"] == 2


def test_local_replica_matches_the_upstream_convention(tmp_path):
    """Upstream counts a row with no prediction as not flagged; the replica must too."""
    benchmark = tmp_path / "bench"
    benchmark.mkdir()
    with open(benchmark / "combined.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tx_hash", "label", "category", "group_incident",
                                               "split_temporal"])
        writer.writeheader()
        writer.writerow({"tx_hash": "0x1", "label": "1", "category": "exploit",
                         "group_incident": "i1", "split_temporal": "test"})
        writer.writerow({"tx_hash": "0x2", "label": "1", "category": "key_compromise",
                         "group_incident": "i2", "split_temporal": "test"})
        writer.writerow({"tx_hash": "0x3", "label": "0", "category": "ordinary",
                         "group_incident": "", "split_temporal": "test"})
    predictions = tmp_path / "pred.csv"
    with open(predictions, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tx_hash", "prediction"])
        writer.writerow(["0x1", "1"])
    result = E.local_replica(benchmark, predictions, split="temporal", track="exploit",
                             base_rate=None)
    assert result["rows"] == 2, "the key-compromise attack is filtered out by the track"
    assert result["without_prediction"] == 1
    assert result["tp"] == 1 and result["fn"] == 0
    assert result["recall"] == 1.0
    assert result["recall_incidents_equal"] == 1.0
