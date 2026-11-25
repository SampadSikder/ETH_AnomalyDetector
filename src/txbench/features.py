"""The two FT-Transformer feature views, with explicit definitions.

Two views exist so that a missing historical call tree cannot silently change
the experiment:

  **tx-receipt**          one row per transaction: the top-level selector
                          candidate plus counts of the receipt logs' first
                          topics. Valid whenever a receipt exists; an empty log
                          list is valid and yields genuine zeros.

  **full-calls-receipt**  one row per transaction: counts of selector
                          candidates over *all* call frames - the outer call
                          included exactly once - plus the same receipt-topic
                          counts. Requires a call tree the source asserts is
                          complete, with input bytes on every frame.

Pipeline rules:
  * sparse counts keyed by token are written first, before any vocabulary;
  * selectors, precompile targets and topics live in separate namespaces;
  * the vocabulary, its frequency cutoff and the scalers are fitted on
    **training rows only**, persisted per split, and out-of-vocabulary mass is
    reported, not dropped silently;
  * identifiers, label evidence, attack categories, source tiers, collection
    paths and detector success flags are excluded from model features;
  * unavailable data stays missing - only an observed absence becomes a zero.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from . import FEATURE_SCHEMA_VERSION
from .extract import (NS_PRECOMPILE, NS_SELECTOR, NS_TOPIC0, full_calls_receipt_tokens,
                      selector_status_summary, tx_receipt_tokens)
from .logs import get_logger
from .trace_model import NormalizedTx

log = get_logger("features")

VIEW_TX_RECEIPT = "tx-receipt"
VIEW_FULL_CALLS = "full-calls-receipt"
VIEWS = (VIEW_TX_RECEIPT, VIEW_FULL_CALLS)

#: Columns that must never be offered to a model as a feature.
NON_FEATURE_COLUMNS = (
    "tx_key", "chain_id", "tx_hash", "record_id", "label", "label_name", "label_status",
    "label_tier", "source_category", "attack_class", "benign_category", "subcategory",
    "behavior_tags", "is_key_compromise", "group_incident", "group_protocol", "group_address",
    "release_id", "is_fixture", "source_file", "source_rows", "provenance_sha256",
    "block_number", "block_timestamp", "partition", "usable", "unusable_reason",
    "upstream_trace_path", "call_tree_status", "receipt_available", "duplicate_count",
    "manifest_schema_version", "view",
)

OOV_COLUMNS = (f"oov_{NS_SELECTOR}_count", f"oov_{NS_TOPIC0}_count", f"oov_{NS_PRECOMPILE}_count")


@dataclass
class ViewRow:
    tx_key: str
    counts: dict[str, int]
    status: dict[str, Any]


@dataclass
class SparseView:
    view: str
    rows: list[ViewRow] = field(default_factory=list)

    def usable_keys(self) -> list[str]:
        return [r.tx_key for r in self.rows if r.status.get("usable")]

    def to_long_frame(self) -> pd.DataFrame:
        records = [{"tx_key": row.tx_key, "token": token, "count": count,
                    "namespace": token.split(":", 1)[0]}
                   for row in self.rows if row.status.get("usable")
                   for token, count in sorted(row.counts.items())]
        return pd.DataFrame(records, columns=["tx_key", "token", "count", "namespace"])

    def to_status_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"tx_key": r.tx_key, **r.status} for r in self.rows])


def build_sparse_view(view: str, txs: Iterable[tuple[str, NormalizedTx | None]],
                      *, missing_reason: str = "no normalised transaction record") -> SparseView:
    """Turn normalised transactions into sparse per-row token counts."""
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; known: {VIEWS}")
    out = SparseView(view=view)
    for tx_key, tx in txs:
        if tx is None:
            out.rows.append(ViewRow(tx_key=tx_key, counts={},
                                    status={"view": view, "usable": False,
                                            "unusable_reason": missing_reason,
                                            "receipt_available": False}))
            continue
        if view == VIEW_TX_RECEIPT:
            counts, status = tx_receipt_tokens(tx)
        else:
            counts, status = full_calls_receipt_tokens(tx)
        status = dict(status)
        status["attempted_calls"] = len(tx.frames)
        status["committed_receipt_events"] = tx.log_count
        status["selector_statuses"] = json.dumps(selector_status_summary(tx), sort_keys=True)
        status["distinct_tokens"] = len(counts)
        status["total_token_count"] = sum(counts.values())
        out.rows.append(ViewRow(tx_key=tx_key, counts=counts, status=status))
    return out


# --- vocabulary ---------------------------------------------------------------

@dataclass
class Vocabulary:
    view: str
    split: str
    min_train_frequency: int
    tokens: list[str]
    document_frequency: dict[str, int]
    fitted_on_rows: int
    fitted_on_keys_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "view": self.view, "split": self.split,
            "min_train_frequency": self.min_train_frequency,
            "fitted_on": "training rows only",
            "fitted_on_rows": self.fitted_on_rows,
            "fitted_on_keys_sha256": self.fitted_on_keys_sha256,
            "tokens": self.tokens,
            "token_count": len(self.tokens),
            "document_frequency": self.document_frequency,
            "namespaces": {NS_SELECTOR: "first four bytes of a normal call's input "
                                        "(a candidate function id, not a verified ABI id)",
                           NS_TOPIC0: "a receipt log's observed first topic (raw_topic0)",
                           NS_PRECOMPILE: "a call to a known precompile address; its first four "
                                          "bytes are arguments, not a function id"},
        }


def fit_vocabulary(sparse: SparseView, train_keys: Iterable[str], *, split: str,
                   min_train_frequency: int = 1) -> Vocabulary:
    """Fit on training rows only. Nothing from validation or test is read."""
    from .hashutil import digest_json
    train = set(train_keys)
    used: list[str] = []
    df_counts: Counter[str] = Counter()
    rows = 0
    for row in sparse.rows:
        if row.tx_key not in train or not row.status.get("usable"):
            continue
        rows += 1
        used.append(row.tx_key)
        for token in row.counts:
            df_counts[token] += 1
    tokens = sorted(t for t, n in df_counts.items() if n >= max(1, int(min_train_frequency)))
    log.info("%s/%s: vocabulary %d tokens from %d training rows (cutoff df>=%d)",
             sparse.view, split, len(tokens), rows, min_train_frequency)
    return Vocabulary(view=sparse.view, split=split, min_train_frequency=min_train_frequency,
                      tokens=tokens,
                      document_frequency={t: df_counts[t] for t in tokens},
                      fitted_on_rows=rows,
                      fitted_on_keys_sha256=digest_json(sorted(used)))


# --- matrices -----------------------------------------------------------------

def build_matrix(sparse: SparseView, vocab: Vocabulary, keys: Iterable[str],
                 labels: Mapping[str, int], partition: str) -> pd.DataFrame:
    """Dense count matrix for one partition, with out-of-vocabulary mass kept.

    A row that is not usable in this view is *not* emitted with zeros: it is
    emitted with `usable=False`, every feature column null, and its reason - so
    the denominator stays honest.
    """
    index = {t: i for i, t in enumerate(vocab.tokens)}
    wanted = [k for k in keys]
    by_key = {r.tx_key: r for r in sparse.rows}
    records: list[dict[str, Any]] = []
    for key in wanted:
        row = by_key.get(key)
        record: dict[str, Any] = {"tx_key": key, "partition": partition,
                                  "label": labels.get(key)}
        usable = bool(row and row.status.get("usable"))
        record["usable"] = usable
        record["unusable_reason"] = None if usable else (
            row.status.get("unusable_reason") if row else "no feature row")
        if not usable:
            for token in vocab.tokens:
                record[token] = None
            for col in OOV_COLUMNS:
                record[col] = None
            records.append(record)
            continue
        vector = [0] * len(vocab.tokens)
        oov = Counter()
        for token, count in row.counts.items():
            position = index.get(token)
            if position is None:
                oov[token.split(":", 1)[0]] += count
            else:
                vector[position] += count
        for token, position in index.items():
            record[token] = vector[position]
        for namespace, col in zip((NS_SELECTOR, NS_TOPIC0, NS_PRECOMPILE), OOV_COLUMNS):
            record[col] = int(oov.get(namespace, 0))
        records.append(record)
    columns = ["tx_key", "partition", "label", "usable", "unusable_reason",
               *vocab.tokens, *OOV_COLUMNS]
    return pd.DataFrame(records, columns=columns)


def fit_scalers(matrix: pd.DataFrame, vocab: Vocabulary) -> dict[str, Any]:
    """Mean / standard deviation per feature column, from training rows only.

    Persisted rather than applied: the training step decides whether to use
    them, and a later split cannot refit them by accident.
    """
    usable = matrix[matrix["usable"] == True]        # noqa: E712 - explicit for clarity
    stats: dict[str, dict[str, float]] = {}
    for column in [*vocab.tokens, *OOV_COLUMNS]:
        series = pd.to_numeric(usable[column], errors="coerce").dropna()
        if not len(series):
            stats[column] = {"mean": 0.0, "std": 0.0, "n": 0}
            continue
        mean = float(series.mean())
        std = float(series.std(ddof=0))
        stats[column] = {"mean": mean, "std": std if std > 0 else 0.0, "n": int(len(series))}
    return {"feature_schema_version": FEATURE_SCHEMA_VERSION, "view": vocab.view,
            "split": vocab.split, "fitted_on": "training rows only",
            "fitted_on_rows": int(len(usable)), "columns": stats}


def coverage(sparse: SparseView, labels: Mapping[str, int] | None = None) -> dict[str, Any]:
    """Per-view coverage. Rows with missing inputs stay in the denominator."""
    total = len(sparse.rows)
    usable = sum(1 for r in sparse.rows if r.status.get("usable"))
    reasons = Counter(str(r.status.get("unusable_reason")) for r in sparse.rows
                      if not r.status.get("usable"))
    by_class: dict[str, dict[str, int]] = {}
    if labels:
        for row in sparse.rows:
            label = labels.get(row.tx_key)
            slot = by_class.setdefault("attack" if label == 1 else
                                       ("negative" if label == 0 else "unlabeled"),
                                       {"eligible": 0, "usable": 0})
            slot["eligible"] += 1
            slot["usable"] += int(bool(row.status.get("usable")))
    return {"view": sparse.view, "eligible_rows": total, "usable_rows": usable,
            "usable_share": round(usable / total, 6) if total else None,
            "unusable_by_reason": dict(reasons), "by_class": by_class}


# --- writing ------------------------------------------------------------------

def feature_dictionary(view: str, vocab: Vocabulary | None) -> dict[str, Any]:
    definitions = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "view": view,
        "row_unit": "one Ethereum mainnet transaction, keyed by (chain_id, tx_hash)",
        "definition": {
            VIEW_TX_RECEIPT: "the top-level call's selector candidate, plus one count per "
                             "distinct first topic over the receipt's logs",
            VIEW_FULL_CALLS: "one count per distinct selector candidate over every call frame "
                             "(the outer call included exactly once), plus one count per "
                             "distinct receipt-log first topic",
        }[view],
        "token_namespaces": {
            f"{NS_SELECTOR}:<4 bytes>": "first four bytes of a normal call's input, of length "
                                        ">= 4 bytes. A candidate function id held as a "
                                        "categorical hex value; not verified against any ABI.",
            f"{NS_TOPIC0}:<32 bytes>": "a receipt log's observed first topic (raw_topic0). No "
                                       "event signature is claimed: an anonymous event may put "
                                       "an argument in that slot, and zero-topic logs exist.",
            f"{NS_PRECOMPILE}:<address>": "a call whose target is a known precompile. Its first "
                                          "four bytes are arguments and are deliberately kept "
                                          "out of the selector namespace.",
        },
        "explicit_statuses": {
            "ok": "a selector was extracted",
            "empty_input": "zero bytes of calldata (e.g. a plain value transfer)",
            "short_input": "one to three bytes of calldata: not a selector",
            "contract_creation": "initcode, not a function call",
            "precompile_target": "arbitrary argument bytes, not a function id",
            "not_a_call": "a frame type that carries no calldata",
            "unavailable": "the source supplied no input bytes for this frame",
        },
        "missingness": "An unavailable input stays null. Only an observed absence - a receipt "
                       "that exists with an empty log list, or a transaction in which no code "
                       "executed - becomes a zero.",
        "non_feature_columns": list(NON_FEATURE_COLUMNS),
        "out_of_vocabulary_columns": list(OOV_COLUMNS),
        "target_column": "label",
        "key_column": "tx_key",
    }
    if vocab is not None:
        definitions["vocabulary"] = {"split": vocab.split, "tokens": len(vocab.tokens),
                                     "min_train_frequency": vocab.min_train_frequency,
                                     "fitted_on_rows": vocab.fitted_on_rows}
    return definitions


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def view_dir(features_root: Path, view: str) -> Path:
    return Path(features_root) / view.replace("-", "_").replace(" ", "_")
