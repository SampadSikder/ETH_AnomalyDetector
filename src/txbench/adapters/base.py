"""The adapter contract: what each detector needs, and what we can prove.

Three kinds of evidence are kept strictly apart, because conflating them is
how a benchmark starts reporting an input as usable when nobody ran anything:

  `schema_check`                a field contract was checked against our data;
  `upstream_preprocess_ok`      a converted record went through the detector's
                                *own* preprocessing entry point, at full
                                fidelity;
  `upstream_preprocess_degraded` the entry point ran, but on a record missing a
                                required field class - a documented deviation,
                                never reported as a pass;
  `paper_reimplementation`      our own adapter, written from the paper;
  `project_implementation`      a component this project owns outright.

Per-transaction verdicts are `usable`, `partial` or `blocked`, and a blocked
record stays in the denominator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..trace_model import NormalizedTx

# evidence levels
SCHEMA_CHECK = "schema_check"
UPSTREAM_OK = "upstream_preprocess_ok"
UPSTREAM_DEGRADED = "upstream_preprocess_degraded"
PAPER_REIMPL = "paper_reimplementation"
PROJECT_IMPL = "project_implementation"

# per-transaction verdicts
USABLE = "usable"
PARTIAL = "partial"
BLOCKED = "blocked"


@dataclass(frozen=True)
class FieldRequirement:
    """One field class a detector needs, and where it would have to come from."""
    name: str
    description: str
    required: bool = True
    #: The data source that could supply it, for the enrichment plan.
    source: str = ""


@dataclass
class DetectorSpec:
    key: str
    name: str
    paper: str
    paper_id: str
    code_repo: str | None = None
    code_revision: str | None = None
    preprocess_entry: str | None = None
    sample_schema: str | None = None
    requirements: tuple[FieldRequirement, ...] = ()
    #: The strongest evidence this project can currently produce, and why.
    attainable_evidence: tuple[str, ...] = (SCHEMA_CHECK,)
    evidence_notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "detector": self.key, "name": self.name,
            "paper": self.paper, "paper_id": self.paper_id,
            "code_repo": self.code_repo, "code_revision": self.code_revision,
            "preprocess_entry": self.preprocess_entry,
            "sample_schema": self.sample_schema,
            "required_fields": [
                {"field": r.name, "required": r.required, "description": r.description,
                 "source": r.source} for r in self.requirements],
            "attainable_evidence": list(self.attainable_evidence),
            "evidence_notes": self.evidence_notes,
        }


@dataclass
class TxCheck:
    """One row of the per-transaction compatibility table."""
    detector: str
    tx_key: str
    verdict: str
    available_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    missing_reasons: dict[str, str] = field(default_factory=dict)
    evidence: str = SCHEMA_CHECK
    notes: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        import json
        return {
            "detector": self.detector, "tx_key": self.tx_key, "verdict": self.verdict,
            "evidence": self.evidence,
            "required_fields_available": ";".join(sorted(self.available_fields)) or None,
            "required_fields_missing": ";".join(sorted(self.missing_fields)) or None,
            "missing_field_reason": json.dumps(self.missing_reasons, sort_keys=True)
            if self.missing_reasons else None,
            "notes": json.dumps(self.notes, sort_keys=True, default=str) if self.notes else None,
        }


#: A checker takes a normalised transaction (or None) and returns a TxCheck.
Checker = Callable[[str, NormalizedTx | None], TxCheck]


def blocked_for_missing_record(detector: str, tx_key: str) -> TxCheck:
    return TxCheck(detector=detector, tx_key=tx_key, verdict=BLOCKED,
                   missing_fields=["record"],
                   missing_reasons={"record": "no normalised transaction record was built for "
                                              "this hash (no upstream trace and nothing cached)"})
