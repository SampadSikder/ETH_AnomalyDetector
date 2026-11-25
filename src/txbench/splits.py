"""Splits: preserve the upstream assignments, audit them, add a validation part.

Rules:
  * every upstream `split_*` column is preserved verbatim;
  * the temporal split is the primary evaluation; the protocol hold-out is
    reported separately; the cross-chain split is metadata only on an
    Ethereum-only cohort (its test side is by definition off-Ethereum);
  * a validation partition is carved out of *training* rows only, by whole
    incident groups and by time, so that no group straddles a boundary and
    validation stays earlier than test;
  * duplicate hashes, incident overlap and chronology are checked and any
    conflict is published as a *new* corrected split, never by editing the
    upstream one.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .logs import get_logger

log = get_logger("splits")

TRAIN, VALIDATION, TEST = "train", "validation", "test"
GROUP_COLUMNS = ("group_incident", "group_protocol", "group_address")

#: What each upstream split actually promises to keep whole. Overlap in a
#: group a split does not claim to hold apart is *by design*, not a leak: a
#: temporal split cuts by date, so the same protocol or sender appears on both
#: sides - that is precisely why the protocol and address splits exist. Overlap
#: is reported for every group column either way; only a broken promise counts
#: as a conflict.
SPLIT_GUARANTEES: dict[str, tuple[str, ...]] = {
    "split_incident": ("group_incident",),
    "split_protocol": ("group_protocol",),
    "split_address": ("group_address",),
    # A temporal split must also keep incidents whole: one incident's
    # transactions share an exploit, so splitting them across the boundary
    # would leak.
    "split_temporal": ("group_incident",),
    "split_cross_chain": (),
}
#: Splits whose definition requires training to end before test begins.
CHRONOLOGICAL_SPLITS = ("split_temporal",)


def guarantees_for(column: str) -> tuple[str, ...]:
    return SPLIT_GUARANTEES.get(column, ("group_incident",))


@dataclass
class SplitAudit:
    split: str
    guaranteed_groups: tuple[str, ...] = ()
    partitions: dict[str, dict[str, int]] = field(default_factory=dict)
    duplicate_hashes: list[str] = field(default_factory=list)
    group_overlap: dict[str, list[str]] = field(default_factory=dict)
    chronology: dict[str, Any] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)
    rows_without_assignment: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"split": self.split, "partitions": self.partitions,
                "guaranteed_groups": list(self.guaranteed_groups),
                "guarantee_held": {g: not self.group_overlap.get(g)
                                   for g in self.guaranteed_groups},
                "rows_without_assignment": self.rows_without_assignment,
                "duplicate_hashes": self.duplicate_hashes[:50],
                "duplicate_hash_count": len(self.duplicate_hashes),
                "group_overlap": {k: v[:25] for k, v in self.group_overlap.items()},
                "group_overlap_counts": {k: len(v) for k, v in self.group_overlap.items()},
                "chronology": self.chronology,
                "conflicts": self.conflicts,
                "clean": not self.conflicts,
                "overlap_by_design": {g: len(v) for g, v in self.group_overlap.items()
                                      if g not in self.guaranteed_groups and v}}


def split_columns(manifest: pd.DataFrame) -> list[str]:
    return sorted(c for c in manifest.columns if c.startswith("split_"))


def audit_split(manifest: pd.DataFrame, column: str) -> SplitAudit:
    audit = SplitAudit(split=column, guaranteed_groups=guarantees_for(column))
    if column not in manifest.columns:
        audit.conflicts.append(f"{column} is not present in the manifest")
        return audit
    frame = manifest[["tx_key", "tx_hash", "label", "block_timestamp", column, *GROUP_COLUMNS]]
    audit.rows_without_assignment = int(frame[column].isna().sum())

    for partition, group in frame.groupby(column, dropna=True):
        labels = Counter(int(v) for v in group["label"])
        audit.partitions[str(partition)] = {
            "rows": int(len(group)), "attacks": labels.get(1, 0), "negatives": labels.get(0, 0)}

    duplicates = frame["tx_hash"].value_counts()
    audit.duplicate_hashes = sorted(str(h) for h, n in duplicates.items() if n > 1)

    for group_column in GROUP_COLUMNS:
        if group_column not in frame.columns:
            continue
        spanning: list[str] = []
        for value, group in frame.dropna(subset=[group_column]).groupby(group_column):
            partitions = {str(p) for p in group[column].dropna().unique()}
            if len({p for p in partitions if p in (TRAIN, TEST, VALIDATION)}) > 1:
                spanning.append(str(value))
        audit.group_overlap[group_column] = sorted(spanning)

    for group_column in audit.guaranteed_groups:
        spanning = audit.group_overlap.get(group_column) or []
        if spanning:
            audit.conflicts.append(
                f"{column} promises to keep {group_column} whole, but {len(spanning)} "
                f"group(s) appear in both train and test")
    if audit.duplicate_hashes:
        audit.conflicts.append(f"{len(audit.duplicate_hashes)} duplicate hash(es) in {column}")

    stamps = frame.dropna(subset=["block_timestamp"])
    train_times = sorted(str(t) for t in stamps.loc[stamps[column] == TRAIN, "block_timestamp"])
    test_times = sorted(str(t) for t in stamps.loc[stamps[column] == TEST, "block_timestamp"])
    if train_times and test_times:
        audit.chronology = {
            "train_first": train_times[0], "train_last": train_times[-1],
            "test_first": test_times[0], "test_last": test_times[-1],
            "train_entirely_before_test": train_times[-1] < test_times[0],
            "overlapping_rows_in_train_after_test_start":
                sum(1 for t in train_times if t >= test_times[0]),
            "chronology_required": column in CHRONOLOGICAL_SPLITS,
        }
        if column in CHRONOLOGICAL_SPLITS and not audit.chronology["train_entirely_before_test"]:
            audit.conflicts.append(
                f"{column} is a temporal split but {audit.chronology['overlapping_rows_in_train_after_test_start']} "
                "training row(s) are dated at or after the first test row")
    return audit


def make_validation_partition(manifest: pd.DataFrame, column: str, *, fraction: float = 0.2,
                              group_column: str = "group_incident",
                              assignments: dict[str, str] | None = None,
                              ) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Carve a validation partition out of the training rows.

    Whole groups move together, and the groups chosen are the *latest* training
    groups by block time, so validation sits between training and test in time
    instead of being sampled across it.
    """
    frame = manifest[["tx_key", "label", "block_timestamp", column, group_column]].copy()
    if assignments:
        # Build the validation split on top of the authoritative assignment
        # (a corrected split when one was needed), not the raw upstream column.
        frame[column] = [assignments.get(k) for k in frame["tx_key"]]
    train = frame[frame[column] == TRAIN]
    if not len(train):
        return (pd.DataFrame(columns=["tx_key", "partition"]),
                {"error": f"{column} has no training rows"})

    keys_by_group: dict[str, list[str]] = defaultdict(list)
    latest: dict[str, str] = {}
    for row in train.itertuples(index=False):
        group = str(getattr(row, group_column) or row.tx_key)
        keys_by_group[group].append(row.tx_key)
        stamp = str(row.block_timestamp or "")
        if stamp > latest.get(group, ""):
            latest[group] = stamp

    ordered = sorted(keys_by_group, key=lambda g: (latest.get(g, ""), g), reverse=True)
    target = int(round(len(train) * max(0.0, min(1.0, fraction))))
    chosen: list[str] = []
    picked_groups: list[str] = []
    for group in ordered:
        if len(chosen) >= target:
            break
        picked_groups.append(group)
        chosen.extend(keys_by_group[group])

    assignment = {key: VALIDATION for key in chosen}
    rows = []
    for row in frame.itertuples(index=False):
        upstream = getattr(row, column)
        if upstream == TRAIN:
            rows.append({"tx_key": row.tx_key,
                         "partition": assignment.get(row.tx_key, TRAIN)})
        elif isinstance(upstream, str) and upstream:
            rows.append({"tx_key": row.tx_key, "partition": upstream})
        else:
            rows.append({"tx_key": row.tx_key, "partition": None})
    out = pd.DataFrame(rows, columns=["tx_key", "partition"])

    joined = out.merge(frame[["tx_key", "label"]], on="tx_key", how="left")
    counts = Counter(p for p in joined["partition"] if p)
    labels = {p: Counter(int(l) for l, q in zip(joined["label"], joined["partition"])
                         if q == p and l is not None) for p in counts}
    validation_stamps = sorted(str(s) for k, s in zip(frame["tx_key"], frame["block_timestamp"])
                               if assignment.get(k) and s)
    meta = {
        "source_split": column,
        "built_on": "corrected assignment" if assignments else "upstream column",
        "method": "latest training groups by block time, whole groups only",
        "group_column": group_column,
        "requested_fraction": fraction,
        "validation_rows": counts.get(VALIDATION, 0),
        "validation_groups": len(picked_groups),
        "realised_fraction": round(counts.get(VALIDATION, 0) / len(train), 6) if len(train) else None,
        "partition_counts": {p: {"rows": n, "attacks": labels[p].get(1, 0),
                                 "negatives": labels[p].get(0, 0)} for p, n in sorted(counts.items())},
        "validation_time_range": ([validation_stamps[0], validation_stamps[-1]]
                                  if validation_stamps else None),
    }
    return out, meta


def corrected_split(manifest: pd.DataFrame, column: str, audit: SplitAudit,
                    new_name: str | None = None) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    """If a split leaks, publish a corrected copy under a new name.

    The upstream column is never modified. Repair rule: a group that straddles
    train and test is moved wholly to the **test** side, so that nothing a
    model trained on can reappear at evaluation time.
    """
    if not audit.conflicts:
        return None, {"source_split": column, "needed": False,
                      "reason": f"{column} holds every guarantee it makes "
                                f"({list(audit.guaranteed_groups)}); overlap in other group "
                                "columns is by this split's design and is reported, not repaired"}
    spanning = {g for col in audit.guaranteed_groups
                for g in (audit.group_overlap.get(col) or [])}
    name = new_name or f"{column}_corrected"
    moved: list[str] = []
    rows = []
    for row in manifest.itertuples(index=False):
        partition = getattr(row, column, None)
        target = partition
        for group_column in audit.guaranteed_groups:
            value = getattr(row, group_column, None)
            if value is not None and str(value) in spanning and partition == TRAIN:
                target = TEST
                moved.append(row.tx_key)
                break
        rows.append({"tx_key": row.tx_key, "partition": target})
    frame = pd.DataFrame(rows, columns=["tx_key", "partition"])
    meta = {"source_split": column, "needed": True, "new_split_name": name,
            "rule": "a group straddling train and test is moved wholly to test",
            "rows_moved_to_test": len(moved), "groups_repaired": sorted(spanning)[:50],
            "group_count_repaired": len(spanning),
            "duplicate_hashes_present": len(audit.duplicate_hashes)}
    return frame, meta


def verify(manifest: pd.DataFrame, *, primary: str, secondary: Iterable[str],
           metadata_only: Iterable[str], validation_fraction: float) -> dict[str, Any]:
    """Audit every upstream split and build the primary split's partitions."""
    columns = split_columns(manifest)
    audits = {column: audit_split(manifest, column) for column in columns}
    primary_column = primary if primary.startswith("split_") else f"split_{primary}"
    secondary_columns = [s if s.startswith("split_") else f"split_{s}" for s in secondary]
    metadata_columns = [s if s.startswith("split_") else f"split_{s}" for s in metadata_only]

    report: dict[str, Any] = {
        "upstream_split_columns": columns,
        "primary": primary_column,
        "secondary_reported_separately": secondary_columns,
        "metadata_only": {
            column: {
                "reason": "an Ethereum-only cohort cannot exercise this split; kept as metadata",
                "partitions": audits[column].partitions if column in audits else {},
            } for column in metadata_columns if column in audits},
        "audits": {column: audit.as_dict() for column, audit in audits.items()},
    }
    if primary_column not in audits:
        report["error"] = f"primary split {primary_column} is absent from the manifest"
        return report

    # Repair first (if the split broke a promise it makes), then carve the
    # validation partition out of whichever assignment is authoritative, so
    # that `primary_with_validation` is always the file to train against.
    corrected_frame, corrected_meta = corrected_split(manifest, primary_column,
                                                      audits[primary_column])
    report["corrected_split"] = corrected_meta
    assignments = None
    if corrected_frame is not None:
        assignments = {row.tx_key: row.partition
                       for row in corrected_frame.itertuples(index=False) if row.partition}
    partitions, validation_meta = make_validation_partition(
        manifest, primary_column, fraction=validation_fraction, assignments=assignments)
    report["validation_partition"] = validation_meta
    report["authoritative_split_file"] = "primary_with_validation.parquet"
    report["_frames"] = {"primary_with_validation": partitions}
    if corrected_frame is not None:
        report["_frames"][corrected_meta["new_split_name"]] = corrected_frame
    else:
        report["_stale_to_remove"] = [f"{primary_column}_corrected.parquet"]
    return report


def write_splits(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Write one parquet per published split, plus the audit report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for stale in report.pop("_stale_to_remove", []):
        path = out_dir / stale
        if path.exists():
            path.unlink()
            log.info("removed stale split artefact %s (no correction is needed)", path.name)
    frames: dict[str, pd.DataFrame] = report.pop("_frames", {})
    for name, frame in frames.items():
        path = out_dir / f"{name}.parquet"
        tmp = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
        written[name] = path
    report_path = out_dir / "splits_report.json"
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(report_path)
    written["report"] = report_path
    return written


def load_partitions(path: Path) -> dict[str, str]:
    frame = pd.read_parquet(path)
    return {row.tx_key: row.partition for row in frame.itertuples(index=False)
            if row.partition}
