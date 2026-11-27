"""Detector runs: the three models this project evaluates on the frozen set.

Each module here produces one *continuous* score per transaction plus an
explicit status for every transaction it could not score, and writes them in
the intake format `txbench.evaluate` enforces. Nothing here reads a label at
scoring time: labels and scores meet only in the evaluator.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

STATUS_SCORED = "scored"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_UNSUPPORTED = "unsupported_input"

#: The columns `txbench.evaluate.read_predictions` requires, in order.
PREDICTION_COLUMNS = ("chain_id", "tx_hash", "model", "version", "split", "status",
                      "score", "score_direction", "verdict", "note")


@dataclass
class PredictionRow:
    """One model's answer about one transaction - a score or a stated failure."""
    tx_key: str
    model: str
    version: str
    split: str
    status: str = STATUS_SCORED
    score: float | None = None
    score_direction: str = "higher_is_attack"
    verdict: int | None = None
    note: str | None = None
    #: Extra, model-specific columns kept in the parquet, never in the CSV.
    extra: dict[str, Any] = field(default_factory=dict)

    def as_csv_row(self) -> dict[str, Any]:
        chain_id, tx_hash = self.tx_key.split(":", 1)
        return {"chain_id": chain_id, "tx_hash": tx_hash, "model": self.model,
                "version": self.version, "split": self.split, "status": self.status,
                "score": "" if self.score is None else repr(float(self.score)),
                "score_direction": self.score_direction,
                "verdict": "" if self.verdict is None else int(self.verdict),
                "note": self.note or ""}

    def as_record(self) -> dict[str, Any]:
        record = {k: v for k, v in asdict(self).items() if k != "extra"}
        record.update(self.extra)
        return record


def write_predictions_csv(rows: Iterable[PredictionRow], path: Path) -> Path:
    """Write the evaluator's intake format, atomically."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_csv_row())
    tmp.replace(path)
    return path


def write_json(obj: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    return path


def load_predictions_csv(path: Path) -> list[dict[str, Any]]:
    with open(Path(path), encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def merge_predictions_csv(paths: Iterable[Path], out_path: Path) -> Path:
    """Concatenate several models' prediction files into one intake file."""
    merged: list[dict[str, Any]] = []
    for path in paths:
        merged.extend(load_predictions_csv(path))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_COLUMNS))
        writer.writeheader()
        writer.writerows(merged)
    tmp.replace(out_path)
    return out_path
