"""What the upstream records already contain, per transaction.

The collector's first job is *not* to fetch. It is to find out which fields a
transaction already has, so the free RPC budget is spent only on what is
genuinely missing. Field-level gaps are preserved, never papered over.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import M_BLOCK_BY_NUMBER, M_RECEIPT, M_TX
from .extract import full_calls_receipt_tokens, tx_receipt_tokens
from .logs import get_logger
from .trace_model import (NormalizedTx, RECEIPT_OK, RECEIPT_PRE_BYZANTIUM, TREE_COMPLETE,
                          TREE_UNAVAILABLE, from_upstream_trace)

log = get_logger("inventory")

#: Envelope fields the FT-Transformer views and the detector adapters rely on.
ENVELOPE_FIELDS = ("from_address", "input_hex", "value_wei", "gas", "block_number")


@dataclass
class TxInventory:
    tx_key: str
    chain_id: int
    tx_hash: str
    upstream_trace_path: Path | None = None
    upstream_present: bool = False
    envelope_complete: bool = False
    missing_envelope_fields: list[str] = field(default_factory=list)
    receipt_present: bool = False
    receipt_status_source: str = "unavailable"
    log_count: int | None = None
    call_tree_status: str = TREE_UNAVAILABLE
    call_tree_frames: int = 0
    call_tree_has_nesting: bool = False
    block_number: int | None = None
    block_hash: str | None = None
    transaction_index: int | None = None
    #: (method, target_kind, target_id) triples still needed from a provider.
    needs: list[tuple[str, str, str]] = field(default_factory=list)
    field_gaps: dict[str, str] = field(default_factory=dict)
    view_tx_receipt_usable: bool = False
    view_full_calls_usable: bool = False
    payload_bytes: int | None = None
    error: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {
            "tx_key": self.tx_key, "chain_id": self.chain_id, "tx_hash": self.tx_hash,
            "upstream_present": self.upstream_present,
            "upstream_trace_path": str(self.upstream_trace_path) if self.upstream_trace_path else None,
            "payload_bytes": self.payload_bytes,
            "envelope_complete": self.envelope_complete,
            "missing_envelope_fields": ",".join(self.missing_envelope_fields) or None,
            "receipt_present": self.receipt_present,
            "receipt_status_source": self.receipt_status_source,
            "log_count": self.log_count,
            "call_tree_status": self.call_tree_status,
            "call_tree_frames": self.call_tree_frames,
            "call_tree_has_nesting": self.call_tree_has_nesting,
            "block_number": self.block_number, "block_hash": self.block_hash,
            "transaction_index": self.transaction_index,
            "needs_methods": ",".join(sorted({m for m, _, _ in self.needs})) or None,
            "needs_count": len(self.needs),
            "field_gaps": json.dumps(self.field_gaps, sort_keys=True) if self.field_gaps else None,
            "view_tx_receipt_usable": self.view_tx_receipt_usable,
            "view_full_calls_usable": self.view_full_calls_usable,
            "error": self.error,
        }


def upstream_trace_path(traces_dir: Path, tx_hash: str) -> Path:
    return Path(traces_dir) / f"{tx_hash}.json"


def load_upstream_trace(traces_dir: Path, tx_hash: str) -> tuple[dict[str, Any] | None, int | None]:
    path = upstream_trace_path(traces_dir, tx_hash)
    if not path.is_file():
        return None, None
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), len(raw)


def normalized_from_upstream(traces_dir: Path, chain_id: int, tx_hash: str,
                             trace_status: str | None = None
                             ) -> tuple[NormalizedTx | None, int | None]:
    raw, size = load_upstream_trace(traces_dir, tx_hash)
    if raw is None:
        return None, None
    return from_upstream_trace(raw, chain_id=chain_id, tx_hash=tx_hash,
                               trace_status=trace_status), size


def take_inventory(tx: NormalizedTx | None, *, chain_id: int, tx_hash: str,
                   traces_dir: Path | None, payload_bytes: int | None = None,
                   verify_upstream: bool = False) -> TxInventory:
    """Decide what, if anything, must be fetched for one transaction."""
    inv = TxInventory(tx_key=f"{chain_id}:{tx_hash}", chain_id=chain_id, tx_hash=tx_hash,
                      upstream_trace_path=(upstream_trace_path(traces_dir, tx_hash)
                                           if traces_dir else None),
                      payload_bytes=payload_bytes)
    if tx is None:
        inv.upstream_present = False
        inv.needs = [(M_TX, "tx", tx_hash), (M_RECEIPT, "tx", tx_hash)]
        inv.field_gaps = {"upstream_trace": "no enriched trace in the release for this hash"}
        inv.error = "upstream_trace_missing"
        return inv

    inv.upstream_present = True
    inv.missing_envelope_fields = [f for f in ENVELOPE_FIELDS if getattr(tx, f, None) is None]
    inv.envelope_complete = not inv.missing_envelope_fields
    inv.receipt_present = tx.logs_available
    inv.receipt_status_source = tx.receipt_status_source
    inv.log_count = tx.log_count
    inv.call_tree_status = tx.call_tree_status
    inv.call_tree_frames = len(tx.frames)
    inv.call_tree_has_nesting = tx.call_tree_has_nesting
    inv.block_number = tx.block_number
    inv.block_hash = tx.block_hash
    inv.transaction_index = tx.transaction_index
    inv.field_gaps = dict(tx.field_gaps)

    _, tx_view = tx_receipt_tokens(tx)
    _, full_view = full_calls_receipt_tokens(tx)
    inv.view_tx_receipt_usable = bool(tx_view["usable"])
    inv.view_full_calls_usable = bool(full_view["usable"])

    needs: list[tuple[str, str, str]] = []
    if verify_upstream or not inv.envelope_complete:
        needs.append((M_TX, "tx", tx_hash))
    # A pre-Byzantium receipt has no status field on chain: fetching it again
    # cannot produce one, so that gap is recorded, not queued.
    if verify_upstream or not inv.receipt_present or \
            inv.receipt_status_source not in (RECEIPT_OK, RECEIPT_PRE_BYZANTIUM):
        needs.append((M_RECEIPT, "tx", tx_hash))
    if tx.block_number is not None and (tx.block_hash is None or tx.block_timestamp is None):
        needs.append((M_BLOCK_BY_NUMBER, "block", str(tx.block_number)))
    inv.needs = needs
    return inv
