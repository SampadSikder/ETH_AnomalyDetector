"""The release's own assessment of each enriched trace.

Two files carry it (step 2 for the attack traces, step 4 for the negatives'),
both keyed by transaction hash. Reading them is what lets this project say a
call tree is *complete* instead of merely *present*, and it is also what
distinguishes a transaction with genuinely no internal calls
(`no_code_executed` - an observed absence, so zero is the right value) from one
whose call tree was never collected.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .logs import get_logger
from .release import Release

log = get_logger("upstream_status")

STATUS_ROLES = ("traces_status_step2", "traces_status_step4")

#: `internal_calls_source` values that mean "no internal calls happened",
#: as opposed to "we could not collect them".
OBSERVED_NO_CALLS = frozenset({"no_code_executed"})


def load_trace_status(rel: Release) -> dict[str, dict[str, Any]]:
    """Merge the release's trace-status files, keyed by lower-case tx hash."""
    merged: dict[str, dict[str, Any]] = {}
    for role in STATUS_ROLES:
        if not rel.has(role):
            continue
        path = rel.path(role)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read %s: %s", role, exc)
            continue
        if not isinstance(data, dict):
            continue
        for tx_hash, record in data.items():
            if not isinstance(record, dict):
                continue
            key = str(tx_hash).strip().lower()
            # step 2 holds the attacks, step 4 the negatives; a hash appearing
            # in both keeps the first (step 2) record and notes the overlap.
            if key in merged:
                merged[key].setdefault("also_in", []).append(role)
                continue
            merged[key] = {"status": str(record.get("status") or "").strip().lower() or None,
                           "internal_calls_source": record.get("internal_calls_source") or None,
                           "internal_calls_exact_gas": record.get("internal_calls_exact_gas"),
                           "notes": record.get("notes") or None,
                           "status_source_file": str(path.relative_to(rel.root))}
    log.info("trace status: %d hashes from %s", len(merged),
             [r for r in STATUS_ROLES if rel.has(r)])
    return merged


def status_for(status_index: dict[str, dict[str, Any]], tx_hash: str) -> str | None:
    record = status_index.get(tx_hash.lower())
    return record.get("status") if record else None


def by_tx_key(status_index: dict[str, dict[str, Any]], chain_id: int) -> dict[str, str]:
    """`{chain_id}:{hash}` -> status, for the collector's planning stage."""
    return {f"{chain_id}:{h}": (r.get("status") or "")
            for h, r in status_index.items() if r.get("status")}
