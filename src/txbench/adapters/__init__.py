"""Per-detector input validation and the compatibility table."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..logs import get_logger
from ..trace_model import NormalizedTx
from .base import BLOCKED, PARTIAL, USABLE, TxCheck
from .detectors import CHECKERS, SPECS

log = get_logger("adapters")

__all__ = ["CHECKERS", "SPECS", "audit", "write_audit", "coverage_table"]


def audit(records: Iterable[tuple[str, NormalizedTx | None]],
          detectors: Iterable[str] | None = None) -> list[TxCheck]:
    """Build the per-transaction compatibility table."""
    keys = list(detectors) if detectors else list(SPECS)
    unknown = [k for k in keys if k not in CHECKERS]
    if unknown:
        raise KeyError(f"unknown detector(s): {unknown}; known: {sorted(SPECS)}")
    checks: list[TxCheck] = []
    for tx_key, tx in records:
        for key in keys:
            checks.append(CHECKERS[key](tx_key, tx))
    return checks


def coverage_table(checks: Iterable[TxCheck], labels: dict[str, int] | None = None
                   ) -> dict[str, Any]:
    """Eligible / validated / blocked counts per detector, with reasons.

    Every record with a missing input stays in the denominator.
    """
    by_detector: dict[str, dict[str, Any]] = {}
    for check in checks:
        slot = by_detector.setdefault(check.detector, {
            "detector": check.detector, "eligible": 0, "usable": 0, "partial": 0, "blocked": 0,
            "evidence": Counter(), "blocking_reasons": Counter(), "by_class": {},
        })
        slot["eligible"] += 1
        slot[check.verdict] += 1
        slot["evidence"][check.evidence] += 1
        for field_name in check.missing_fields:
            slot["blocking_reasons"][field_name] += 1
        if labels:
            label = labels.get(check.tx_key)
            name = "attack" if label == 1 else ("negative" if label == 0 else "unlabeled")
            cell = slot["by_class"].setdefault(name, {"eligible": 0, "usable": 0,
                                                      "partial": 0, "blocked": 0})
            cell["eligible"] += 1
            cell[check.verdict] += 1
    out = {}
    for key, slot in sorted(by_detector.items()):
        spec = SPECS.get(key)
        out[key] = {
            **{k: v for k, v in slot.items() if k not in ("evidence", "blocking_reasons")},
            "evidence_levels": dict(slot["evidence"]),
            "missing_field_counts": dict(slot["blocking_reasons"].most_common()),
            "usable_share": (round(slot["usable"] / slot["eligible"], 6)
                             if slot["eligible"] else None),
            "attainable_evidence": list(spec.attainable_evidence) if spec else [],
            "evidence_notes": spec.evidence_notes if spec else "",
        }
    return out


def enrichment_plan(checks: Iterable[TxCheck]) -> dict[str, Any]:
    """The hashes and the exact missing profile needed to unlock each detector."""
    plan: dict[str, Any] = {}
    for check in checks:
        if check.verdict == USABLE:
            continue
        slot = plan.setdefault(check.detector, {"detector": check.detector,
                                                "missing_profile": {}, "tx_count": 0,
                                                "example_tx_keys": []})
        slot["tx_count"] += 1
        if len(slot["example_tx_keys"]) < 25:
            slot["example_tx_keys"].append(check.tx_key)
        for field_name, reason in check.missing_reasons.items():
            entry = slot["missing_profile"].setdefault(field_name, {"reason": reason, "count": 0,
                                                                    "source": None})
            entry["count"] += 1
    for key, slot in plan.items():
        spec = SPECS.get(key)
        if not spec:
            continue
        sources = {r.name: r.source for r in spec.requirements}
        for field_name, entry in slot["missing_profile"].items():
            entry["source"] = sources.get(field_name)
    return plan


def write_audit(checks: list[TxCheck], out_dir: Path, *, labels: dict[str, int] | None = None,
                extras: dict[str, Any] | None = None) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    table = pd.DataFrame([c.as_row() for c in checks])
    table_path = out_dir / "compatibility_table.parquet"
    tmp = table_path.with_suffix(".parquet.tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(table_path)
    written["compatibility_table"] = table_path

    report = {
        "detectors": {k: spec.as_dict() for k, spec in sorted(SPECS.items())},
        "coverage": coverage_table(checks, labels),
        "enrichment_plan": enrichment_plan(checks),
        "evidence_levels": {
            "schema_check": "a field contract was checked against this project's data",
            "upstream_preprocess_ok": "a converted record ran through the detector's own "
                                      "preprocessing entry point at full fidelity",
            "upstream_preprocess_degraded": "the entry point ran, but on a record missing a "
                                            "required field class (documented deviation)",
            "paper_reimplementation": "this project's adapter, written from the paper",
            "project_implementation": "a component this project owns outright",
        },
    }
    if extras:
        report.update(extras)
    report_path = out_dir / "adapter_report.json"
    tmp = report_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(report_path)
    written["adapter_report"] = report_path
    return written
