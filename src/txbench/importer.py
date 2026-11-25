"""Import the upstream labels and freeze an immutable, auditable cohort.

Hard rules, enforced here and asserted by the tests:
  * labels are imported exactly as supplied; nothing is inferred;
  * an unknown label, a validation failure or a conflicting duplicate NEVER
    becomes a benign row - it is quarantined and counted;
  * a label applies to a whole transaction;
  * every filter reports its row count before and after;
  * the same source files and config produce the same sorted hashes and labels.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from . import MANIFEST_SCHEMA_VERSION, __version__
from .errors import ValidationError
from .hashutil import cohort_digest, digest_json
from .ids import chain_id_for_slug, normalize_chain_id, normalize_tx_hash, tx_key
from .logs import get_logger
from .release import Release
from . import schema_map as sm

log = get_logger("importer")

#: Fields whose equality decides whether two rows for one hash are the "same
#: record". Provenance-only fields (source row, file) are excluded.
IDENTITY_FIELDS = ("chain_id", "tx_hash", "label", "source_category", "subcategory",
                   "label_tier", "group_incident", "group_protocol", "group_address")

#: The columns whose digest pins the cohort's identity.
DIGEST_FIELDS = ("tx_key", "label")

MANIFEST_COLUMNS = [
    "tx_key", "chain_id", "tx_hash", "label", "label_name", "label_status",
    "source_category", "attack_class", "benign_category", "subcategory",
    "behavior_tags", "label_tier", "is_key_compromise",
    "group_incident", "group_protocol", "group_address",
    "block_number", "block_timestamp",
    "release_id", "is_fixture", "source_file", "source_rows", "duplicate_count",
    "record_id", "provenance_sha256", "manifest_schema_version",
]


@dataclass
class FilterStep:
    name: str
    rows_in: int
    rows_out: int
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def dropped(self) -> int:
        return self.rows_in - self.rows_out

    def as_dict(self) -> dict[str, Any]:
        return {"filter": self.name, "rows_in": self.rows_in, "rows_out": self.rows_out,
                "dropped": self.dropped, **({"detail": self.detail} if self.detail else {})}


@dataclass
class ImportResult:
    manifest: pd.DataFrame
    quarantine: pd.DataFrame
    steps: list[FilterStep]
    mapping: sm.SchemaMapping
    meta: dict[str, Any]

    @property
    def cohort_sha256(self) -> str:
        return self.meta["cohort_sha256"]


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValidationError(f"{path} has no header row")
        rows = [dict(r) for r in reader]
        return list(reader.fieldnames), rows


def _normalize_row(raw: dict[str, str], mapping: sm.SchemaMapping, row_number: int,
                   rel: Release) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return (record, quarantine_entry). Exactly one is non-None."""
    def bad(reason: str, detail: str = "") -> tuple[None, dict[str, Any]]:
        return None, {
            # Always text: duplicate quarantine rows carry a comma-joined list,
            # so a mixed int/str column would break the parquet schema.
            "source_row": str(row_number),
            "source_file": mapping.source_file,
            "reason": reason,
            "detail": detail,
            "raw_tx_hash": (raw.get(mapping.resolved.get("tx_hash", "")) or "")[:80],
            "raw_chain_id": (raw.get(mapping.resolved.get("chain_id", "")) or "")[:20],
            "raw_label": (raw.get(mapping.resolved.get("label", "")) or "")[:20],
            "release_id": rel.release_id,
            "is_fixture": rel.is_fixture,
        }

    try:
        chain_id = normalize_chain_id(mapping.get(raw, "chain_id"))
    except ValidationError as exc:
        return bad("invalid_chain_id", str(exc))
    try:
        tx_hash = normalize_tx_hash(mapping.get(raw, "tx_hash"))
    except ValidationError as exc:
        return bad("invalid_tx_hash", str(exc))

    slug = mapping.get(raw, "chain_slug")
    slug_id = chain_id_for_slug(slug)
    if slug_id is not None and slug_id != chain_id:
        return bad("chain_slug_id_disagreement", f"slug={slug!r} -> {slug_id}, chain_id={chain_id}")

    label, label_status = sm.parse_label(mapping.get(raw, "label"))
    if label is None:
        return bad(f"label_{label_status}", f"label={mapping.get(raw, 'label')!r}")

    category = mapping.get(raw, "source_category")
    is_attack = label == 1
    attack_class = category if (is_attack and category in sm.ATTACK_CLASSES) else None
    if is_attack and attack_class is None:
        # An attack row whose class is not one of the documented classes is an
        # upstream schema surprise: quarantine instead of guessing.
        return bad("unknown_attack_class", f"category={category!r}")

    block_number = mapping.get(raw, "block_number")
    try:
        block_number_i = int(block_number) if block_number is not None else None
    except ValueError:
        return bad("invalid_block_number", f"block_number={block_number!r}")

    record: dict[str, Any] = {
        "tx_key": tx_key(chain_id, tx_hash),
        "chain_id": chain_id,
        "tx_hash": tx_hash,
        "label": label,
        "label_name": mapping.get(raw, "label_name"),
        "label_status": label_status,
        "source_category": category,
        "attack_class": attack_class,
        "benign_category": None if is_attack else category,
        "subcategory": mapping.get(raw, "subcategory"),
        "behavior_tags": sm.LIST_SEPARATOR.join(sm.parse_list(mapping.get(raw, "behavior_tags"))),
        "label_tier": mapping.get(raw, "label_tier"),
        "is_key_compromise": attack_class == "key_compromise",
        "group_incident": mapping.get(raw, "group_incident"),
        "group_protocol": mapping.get(raw, "group_protocol"),
        "group_address": mapping.get(raw, "group_address"),
        "block_number": block_number_i,
        "block_timestamp": mapping.get(raw, "block_timestamp"),
        "release_id": rel.release_id,
        "is_fixture": rel.is_fixture,
        "source_file": mapping.source_file,
        "source_rows": str(row_number),
        "duplicate_count": 1,
        "record_id": mapping.get(raw, "record_id"),
    }
    for col in mapping.split_columns:
        record[col] = (raw.get(col) or "").strip() or None
    return record, None


def _collapse_duplicates(records: list[dict[str, Any]],
                         ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Collapse identical duplicates; quarantine conflicting ones."""
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_key[rec["tx_key"]].append(rec)

    kept: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    stats = {"unique_keys": len(by_key), "identical_collapsed": 0, "conflicting_keys": 0,
             "conflicting_rows": 0}

    for key, group in by_key.items():
        if len(group) == 1:
            kept.append(group[0])
            continue
        fingerprints = {digest_json([rec.get(f) for f in IDENTITY_FIELDS]) for rec in group}
        if len(fingerprints) == 1:
            merged = dict(group[0])
            merged["source_rows"] = ",".join(sorted((r["source_rows"] for r in group), key=_int_key))
            merged["duplicate_count"] = len(group)
            kept.append(merged)
            stats["identical_collapsed"] += len(group) - 1
        else:
            stats["conflicting_keys"] += 1
            stats["conflicting_rows"] += len(group)
            for rec in group:
                quarantined.append({
                    "source_row": rec["source_rows"],
                    "source_file": rec["source_file"],
                    "reason": "conflicting_duplicate",
                    "detail": json.dumps({f: rec.get(f) for f in IDENTITY_FIELDS}, sort_keys=True,
                                         default=str),
                    "raw_tx_hash": rec["tx_hash"],
                    "raw_chain_id": str(rec["chain_id"]),
                    "raw_label": str(rec["label"]),
                    "release_id": rec["release_id"],
                    "is_fixture": rec["is_fixture"],
                })
    return kept, quarantined, stats


def _int_key(text: str) -> tuple[int, str]:
    try:
        return int(text), ""
    except ValueError:
        return 1 << 62, text


def import_labels(rel: Release, *, chain_id: int, role: str = "combined") -> ImportResult:
    """Read the release's benchmark table into a frozen cohort manifest."""
    source = rel.path(role)
    rel_source = str(source.relative_to(rel.root))
    columns, raw_rows = _read_csv_rows(source)
    mapping = sm.resolve(columns, rel_source)
    log.info("%s: %d rows, %d columns; split columns: %s",
             rel_source, len(raw_rows), len(columns), mapping.split_columns)

    steps: list[FilterStep] = [FilterStep("read_source", len(raw_rows), len(raw_rows),
                                          {"columns": len(columns)})]

    # 1. validate ids / hashes / labels
    records: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for i, raw in enumerate(raw_rows, start=1):
        rec, bad = _normalize_row(raw, mapping, i, rel)
        if rec is not None:
            records.append(rec)
        else:
            quarantine.append(bad)
            reasons[bad["reason"]] += 1
    steps.append(FilterStep("validate_ids_and_labels", len(raw_rows), len(records),
                            {"quarantined_by_reason": dict(reasons)}))

    # 2. chain filter
    before = len(records)
    other_chains = Counter(r["chain_id"] for r in records if r["chain_id"] != chain_id)
    records = [r for r in records if r["chain_id"] == chain_id]
    steps.append(FilterStep(f"chain_id_eq_{chain_id}", before, len(records),
                            {"rows_by_other_chain_id": {str(k): v for k, v in sorted(other_chains.items())}}))

    # 3. duplicates
    before = len(records)
    records, dup_quarantine, dup_stats = _collapse_duplicates(records)
    quarantine.extend(dup_quarantine)
    steps.append(FilterStep("collapse_duplicates", before, len(records), dup_stats))

    records.sort(key=lambda r: r["tx_key"])
    manifest = _to_frame(records, mapping)
    quarantine_df = pd.DataFrame(quarantine, columns=[
        "source_row", "source_file", "reason", "detail", "raw_tx_hash", "raw_chain_id",
        "raw_label", "release_id", "is_fixture"]) if quarantine else pd.DataFrame(columns=[
        "source_row", "source_file", "reason", "detail", "raw_tx_hash", "raw_chain_id",
        "raw_label", "release_id", "is_fixture"])

    label_counts = Counter(int(v) for v in manifest["label"])
    # pandas renders a missing object value as NaN, which is truthy: drop first.
    class_counts = Counter(str(v) for v in manifest["attack_class"].dropna() if str(v))
    benign_counts = Counter(str(v) for v in manifest["benign_category"].dropna() if str(v))
    meta = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "txbench_tool_version": __version__,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "chain_id": chain_id,
        "release": rel.provenance(),
        "schema_mapping": mapping.as_dict(),
        "filters": [s.as_dict() for s in steps],
        "rows": int(len(manifest)),
        "label_counts": {str(k): v for k, v in sorted(label_counts.items())},
        "attack_class_counts": dict(sorted(class_counts.items())),
        "benign_category_counts": dict(sorted(benign_counts.items())),
        "quarantined_rows": int(len(quarantine_df)),
        "quarantine_reasons": dict(Counter(q["reason"] for q in quarantine)),
        "upstream_split_columns": sorted(mapping.split_columns),
        "cohort_sha256": cohort_digest(records, DIGEST_FIELDS),
        "cohort_digest_fields": list(DIGEST_FIELDS),
        "is_fixture": rel.is_fixture,
    }
    log.info("cohort: %d rows (%s attacks, %s negatives); digest %s",
             len(manifest), label_counts.get(1, 0), label_counts.get(0, 0),
             meta["cohort_sha256"][:16])
    return ImportResult(manifest=manifest, quarantine=quarantine_df, steps=steps,
                        mapping=mapping, meta=meta)


def _to_frame(records: Iterable[dict[str, Any]], mapping: sm.SchemaMapping) -> pd.DataFrame:
    cols = MANIFEST_COLUMNS + sorted(mapping.split_columns)
    df = pd.DataFrame(list(records), columns=cols)
    if not len(df):
        return df
    df["chain_id"] = df["chain_id"].astype("int32")
    df["label"] = df["label"].astype("int8")
    df["is_key_compromise"] = df["is_key_compromise"].astype(bool)
    df["is_fixture"] = df["is_fixture"].astype(bool)
    df["duplicate_count"] = df["duplicate_count"].astype("int32")
    df["block_number"] = df["block_number"].astype("Int64")
    df["manifest_schema_version"] = MANIFEST_SCHEMA_VERSION
    df["provenance_sha256"] = [
        digest_json({"release": r.release_id if hasattr(r, "release_id") else r,
                     "source_file": sf, "source_rows": sr, "tx_key": k})
        for r, sf, sr, k in zip(df["release_id"], df["source_file"], df["source_rows"], df["tx_key"])
    ]
    return df.loc[:, cols]


def write_manifest(result: ImportResult, *, manifest_path: Path, meta_path: Path,
                   mapping_path: Path, quarantine_path: Path) -> dict[str, Path]:
    """Write the manifest and its sidecars atomically. Never edits in place."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    _atomic_parquet(result.manifest, manifest_path)
    written["manifest"] = manifest_path
    _atomic_parquet(result.quarantine, quarantine_path)
    written["quarantine"] = quarantine_path
    _atomic_text(mapping_path, json.dumps(result.mapping.as_dict(), indent=2, sort_keys=True) + "\n")
    written["schema_mapping"] = mapping_path
    meta = dict(result.meta)
    meta["artifacts"] = {k: str(v) for k, v in written.items()}
    _atomic_text(meta_path, json.dumps(meta, indent=2, sort_keys=True, default=str) + "\n")
    written["manifest_meta"] = meta_path
    return written


def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "tx_key" not in df.columns:
        raise ValidationError(f"{path} is not a cohort manifest (no tx_key column)")
    return df
