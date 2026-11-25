"""Evaluation tasks, and the reproducible mapping that defines each one.

Three tasks:

  `exploit`            the main task: contract exploits against the
                       evidence-backed negatives. Key compromises are excluded
                       from the positives and reported separately, because a
                       stolen key leaves no on-chain weakness to detect.
  `key_compromise`     the same negatives, with key-compromise attacks as the
                       positives. Reported on its own, never folded into the
                       main number.
  `price_manipulation` a five-detector family task. Membership comes from a
                       documented, re-runnable mapping over the release's own
                       vulnerability-type and SCWE fields. A record the rules
                       cannot resolve is marked `ambiguous` and is left out of
                       both the positives and the negatives - not quietly
                       resolved either way.

The legitimate control hashes of the price-manipulation task are frozen: the
hash list and its digest are written with the task so a later run cannot
redefine the controls.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .hashutil import cohort_digest, digest_json
from .ids import normalize_tx_hash
from .logs import get_logger

log = get_logger("tasks")

TASK_EXPLOIT = "exploit"
TASK_KEY_COMPROMISE = "key_compromise"
TASK_PRICE_MANIPULATION = "price_manipulation"

#: Rules over the release's `vuln_type` and `scwe_titles` fields. Order does
#: not matter: a record matching any `ambiguous` rule stays ambiguous even if
#: it also matches an `include` rule, because the conflict is the ambiguity.
PRICE_MANIPULATION_RULES: dict[str, dict[str, list[str]]] = {
    "include": {
        "vuln_type": [
            r"^vuln/oracle/price-manipulation$",
            r"^vuln/oracle/spot-price$",
            r"^vuln/oracle/stale-price$",
            r"^vuln/logic/price-calculation$",
            r"^Vulnerable Price Dependency$",
        ],
        "scwe_titles": [
            r"Price Oracle Manipulation",
            r"Reliance on Low-Liquidity Spot Prices",
        ],
    },
    "ambiguous": {
        "vuln_type": [
            r"^vuln/defi/sandwich-attack$",       # ordering, not a mispriced oracle read
            r"^vuln/defi/slippage$",              # a missing guard, not a manipulated price
            r"^vuln/arithmetic/rounding$",
        ],
        "scwe_titles": [
            r"Missing Slippage Protection",
            r"Extractable Value from Predictable Transaction Ordering",
            r"ERC4626 Share Inflation via Donations",   # share price, but by donation
            r"Inconsistent Rounding Direction",
        ],
    },
}

MEMBER_INCLUDED = "included"
MEMBER_AMBIGUOUS = "ambiguous"
MEMBER_EXCLUDED = "excluded"


@dataclass
class TaskDefinition:
    name: str
    positives: list[str] = field(default_factory=list)
    negatives: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    rules: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    detectors: tuple[str, ...] = ()

    @property
    def eligible(self) -> list[str]:
        return sorted(set(self.positives) | set(self.negatives))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.name,
            "detectors": list(self.detectors),
            "positives": len(self.positives),
            "negatives": len(self.negatives),
            "ambiguous_left_unresolved": len(self.ambiguous),
            "eligible_hashes": len(self.eligible),
            "rules": self.rules,
            "notes": self.notes,
            "positive_keys_sha256": digest_json(sorted(self.positives)),
            "control_keys_sha256": digest_json(sorted(self.negatives)),
            "ambiguous_keys_sha256": digest_json(sorted(self.ambiguous)),
        }


def _matches(value: str | None, patterns: Iterable[str]) -> list[str]:
    if not value:
        return []
    return [p for p in patterns if re.search(p, value)]


def load_vulnerability_index(accepted_attacks_csv: Path, chain_slug: str = "eth"
                             ) -> dict[str, dict[str, str]]:
    """Read the release's accepted-attack table, keyed by lower-case tx hash."""
    index: dict[str, dict[str, str]] = {}
    with open(accepted_attacks_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if chain_slug and (row.get("chain") or "").strip().lower() != chain_slug:
                continue
            try:
                tx_hash = normalize_tx_hash(row.get("tx_hash"))
            except Exception:                            # noqa: BLE001 - skip unusable rows
                continue
            index[tx_hash] = {
                "vuln_type": (row.get("vuln_type") or "").strip(),
                "vuln_type_blocksec": (row.get("vuln_type_blocksec") or "").strip(),
                "vuln_type_registry": (row.get("vuln_type_registry") or "").strip(),
                "scwe_titles": (row.get("scwe_titles") or "").strip(),
                "scwe_ids": (row.get("scwe_ids") or "").strip(),
                "attack_class": (row.get("attack_class") or "").strip(),
            }
    return index


def classify_price_manipulation(record: dict[str, str]) -> tuple[str, dict[str, list[str]]]:
    """Apply the documented rules to one accepted-attack record."""
    hits: dict[str, list[str]] = {}
    for decision in ("include", "ambiguous"):
        for field_name, patterns in PRICE_MANIPULATION_RULES[decision].items():
            matched = _matches(record.get(field_name), patterns)
            if matched:
                hits[f"{decision}:{field_name}"] = matched
    ambiguous = any(k.startswith("ambiguous:") for k in hits)
    included = any(k.startswith("include:") for k in hits)
    if ambiguous:
        return MEMBER_AMBIGUOUS, hits
    if included:
        return MEMBER_INCLUDED, hits
    return MEMBER_EXCLUDED, hits


def build_tasks(manifest: pd.DataFrame, *, vulnerability_index: dict[str, dict[str, str]] | None,
                detectors_all: tuple[str, ...] = ("blockgpt", "blockscan", "blocklens",
                                                  "ft_transformer"),
                detectors_price: tuple[str, ...] = ("blockgpt", "blockscan", "blocklens",
                                                    "defiguard", "ft_transformer"),
                ) -> dict[str, TaskDefinition]:
    negatives = sorted(manifest.loc[manifest["label"] == 0, "tx_key"])
    exploits = sorted(manifest.loc[(manifest["label"] == 1) &
                                   (manifest["attack_class"] == "exploit"), "tx_key"])
    key_compromises = sorted(manifest.loc[(manifest["label"] == 1) &
                                          (manifest["attack_class"] == "key_compromise"), "tx_key"])

    tasks: dict[str, TaskDefinition] = {
        TASK_EXPLOIT: TaskDefinition(
            name=TASK_EXPLOIT, positives=exploits, negatives=negatives,
            rules={"positives": "manifest attack_class == 'exploit'",
                   "negatives": "every evidence-backed negative in the cohort",
                   "excluded": "key-compromise attacks, reported as their own task"},
            notes="The main task. Key compromises are excluded from the positives because a "
                  "stolen key leaves no contract-level weakness for a detector to find.",
            detectors=detectors_all),
        TASK_KEY_COMPROMISE: TaskDefinition(
            name=TASK_KEY_COMPROMISE, positives=key_compromises, negatives=negatives,
            rules={"positives": "manifest attack_class == 'key_compromise'",
                   "negatives": "every evidence-backed negative in the cohort"},
            notes="Reported separately and never merged into the main result.",
            detectors=detectors_all),
    }

    pm_positives: list[str] = []
    pm_ambiguous: list[str] = []
    decisions: dict[str, Any] = {}
    unmapped = 0
    if vulnerability_index:
        for row in manifest.loc[manifest["label"] == 1].itertuples(index=False):
            record = vulnerability_index.get(row.tx_hash)
            if record is None:
                unmapped += 1
                continue
            decision, hits = classify_price_manipulation(record)
            decisions[row.tx_key] = {"decision": decision, "matched": hits,
                                     "vuln_type": record.get("vuln_type"),
                                     "scwe_titles": record.get("scwe_titles")}
            if decision == MEMBER_INCLUDED:
                pm_positives.append(row.tx_key)
            elif decision == MEMBER_AMBIGUOUS:
                pm_ambiguous.append(row.tx_key)
    tasks[TASK_PRICE_MANIPULATION] = TaskDefinition(
        name=TASK_PRICE_MANIPULATION, positives=sorted(pm_positives), negatives=negatives,
        ambiguous=sorted(pm_ambiguous),
        rules={"source_file": "step3_oracle/output/attack_txs_accepted.csv",
               "fields_read": ["vuln_type", "scwe_titles"],
               "patterns": PRICE_MANIPULATION_RULES,
               "conflict_rule": "a record matching any ambiguous pattern stays ambiguous, even "
                                "if it also matches an include pattern",
               "attacks_without_a_source_record": unmapped,
               "controls": "every evidence-backed negative in the cohort, frozen by hash"},
        notes="Membership is derived from the release's own vulnerability-type and SCWE fields "
              "by the patterns above. Ambiguous records are left in neither class. The control "
              "hashes are frozen with this definition.",
        detectors=detectors_price)
    tasks[TASK_PRICE_MANIPULATION].rules["decisions_written_to"] = \
        "price_manipulation_decisions.parquet"
    log.info("tasks: exploit %d+, key_compromise %d+, price_manipulation %d+ (%d ambiguous), "
             "%d negatives", len(exploits), len(key_compromises), len(pm_positives),
             len(pm_ambiguous), len(negatives))
    return tasks, decisions


def write_tasks(tasks: dict[str, TaskDefinition], decisions: dict[str, Any], out_dir: Path,
                ) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, task in tasks.items():
        rows = ([{"tx_key": k, "role": "positive"} for k in task.positives] +
                [{"tx_key": k, "role": "control"} for k in task.negatives] +
                [{"tx_key": k, "role": "ambiguous_unresolved"} for k in task.ambiguous])
        frame = pd.DataFrame(rows, columns=["tx_key", "role"]).sort_values(["role", "tx_key"])
        path = out_dir / f"task_{name}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
        written[name] = path
    if decisions:
        frame = pd.DataFrame(
            [{"tx_key": k, "decision": v["decision"], "vuln_type": v.get("vuln_type"),
              "scwe_titles": v.get("scwe_titles"),
              "matched_patterns": json.dumps(v["matched"], sort_keys=True)}
             for k, v in sorted(decisions.items())])
        path = out_dir / "price_manipulation_decisions.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(path)
        written["price_manipulation_decisions"] = path
    summary = {name: task.as_dict() for name, task in sorted(tasks.items())}
    summary["frozen_control_digest"] = cohort_digest(
        [{"tx_key": k} for k in tasks[TASK_PRICE_MANIPULATION].negatives], ("tx_key",))
    path = out_dir / "tasks.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    tmp.replace(path)
    written["tasks"] = path
    return written
