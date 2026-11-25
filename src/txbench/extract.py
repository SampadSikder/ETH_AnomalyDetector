"""Byte-level extraction with explicit, auditable definitions.

Two things are extracted, and only these two carry into the tabular feature
views:

  * **selector candidates** - the first four bytes of a *normal call's* input
    when that input is at least four bytes long. Four bytes of calldata are a
    candidate function id and nothing more: nothing here verifies them against
    an ABI, so they stay categorical hex values and are never renamed to a
    function signature.
  * **receipt log first topics** - the observed first topic of each receipt
    log, stored as `raw_topic0`. A non-anonymous ABI event ordinarily uses
    topic 0 for its signature hash, but an anonymous event may use that slot
    for an argument, and zero-topic logs exist, so no event signature is
    claimed from the bytes alone.

Everything that is not a four-byte selector gets an explicit status instead of
a silent absence: contract creation, short input, empty input, a precompile
target, a frame type that carries no calldata, or unavailable bytes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .trace_model import (CREATE_TYPES, CALL_TYPES_WITH_INPUT, CallFrame, NormalizedTx,
                          TREE_COMPLETE, ReceiptLog)

# --- selector statuses --------------------------------------------------------
SEL_OK = "ok"                            # >= 4 bytes of calldata on a normal call
SEL_EMPTY_INPUT = "empty_input"          # 0 bytes (e.g. a plain value transfer)
SEL_SHORT_INPUT = "short_input"          # 1-3 bytes: not a selector
SEL_CONTRACT_CREATION = "contract_creation"   # initcode, not a function call
SEL_PRECOMPILE = "precompile_target"     # bytes are precompile arguments
SEL_NOT_A_CALL = "not_a_call"            # SELFDESTRUCT and friends
SEL_UNAVAILABLE = "unavailable"          # the source supplied no input bytes

#: Token namespaces. Selectors and topics never share a vocabulary slot, and a
#: precompile's argument bytes never enter the selector namespace.
NS_SELECTOR = "sel"
NS_TOPIC0 = "tpc0"
NS_PRECOMPILE = "pre"

#: Known mainnet precompile addresses (ecrecover .. BLS12-381 / KZG).
PRECOMPILES = frozenset(f"0x{i:040x}" for i in range(0x01, 0x12))


@dataclass(frozen=True)
class SelectorCandidate:
    """The outcome of looking for a selector in one call frame."""
    frame_index: int
    call_type: str
    status: str
    selector: str | None = None        # set only when status == SEL_OK
    #: First four bytes of a precompile call's input, kept out of the selector
    #: namespace but not thrown away.
    candidate_bytes: str | None = None
    to_address: str | None = None
    is_root: bool = False
    reverted: bool | None = None

    @property
    def token(self) -> str | None:
        if self.status == SEL_OK and self.selector:
            return f"{NS_SELECTOR}:{self.selector}"
        if self.status == SEL_PRECOMPILE and self.to_address:
            return f"{NS_PRECOMPILE}:{self.to_address}"
        return None


def input_byte_length(input_hex: str | None) -> int | None:
    if input_hex is None:
        return None
    body = input_hex[2:] if input_hex.startswith("0x") else input_hex
    return len(body) // 2


def selector_from_input(input_hex: str | None) -> tuple[str | None, str]:
    """The core rule: first four bytes of an input of at least four bytes."""
    length = input_byte_length(input_hex)
    if length is None:
        return None, SEL_UNAVAILABLE
    if length == 0:
        return None, SEL_EMPTY_INPUT
    if length < 4:
        return None, SEL_SHORT_INPUT
    return "0x" + input_hex[2:10].lower(), SEL_OK


def frame_selector(frame: CallFrame) -> SelectorCandidate:
    """Apply the rule to one frame, honouring call type and precompiles."""
    call_type = (frame.call_type or "UNKNOWN").upper()
    common = {"frame_index": frame.index, "call_type": call_type,
              "to_address": frame.to_address, "is_root": frame.is_root,
              "reverted": frame.reverted}

    if call_type in CREATE_TYPES or (frame.is_root and frame.to_address is None):
        return SelectorCandidate(status=SEL_CONTRACT_CREATION, **common)
    if call_type not in CALL_TYPES_WITH_INPUT and call_type != "UNKNOWN":
        return SelectorCandidate(status=SEL_NOT_A_CALL, **common)

    selector, status = selector_from_input(frame.input_hex)
    if frame.to_address in PRECOMPILES:
        # A precompile takes raw arguments, not an ABI-encoded call: the first
        # four bytes are arbitrary data and must not be presented as a
        # function id.
        return SelectorCandidate(status=SEL_PRECOMPILE,
                                 candidate_bytes=selector if status == SEL_OK else None,
                                 **common)
    return SelectorCandidate(status=status, selector=selector, **common)


def top_level_selector(tx: NormalizedTx) -> SelectorCandidate:
    """The selector candidate of the transaction's own top-level call."""
    root = next((f for f in tx.frames if f.is_root), None)
    if root is None:
        return SelectorCandidate(frame_index=0, call_type="UNKNOWN", status=SEL_UNAVAILABLE)
    if tx.is_contract_creation:
        return SelectorCandidate(frame_index=root.index, call_type=root.call_type,
                                 status=SEL_CONTRACT_CREATION, to_address=root.to_address,
                                 is_root=True, reverted=root.reverted)
    return frame_selector(root)


def all_frame_selectors(tx: NormalizedTx) -> list[SelectorCandidate]:
    """Selector candidates for every frame, with the outer call exactly once.

    `tx.frames` always holds the root at index 0 and never holds it twice:
    `trace_model` removes a duplicated root when a source supplies one, and
    records that it did.
    """
    return [frame_selector(f) for f in tx.frames]


# --- receipt topics -----------------------------------------------------------

@dataclass(frozen=True)
class TopicObservation:
    log_index: int | None
    address: str | None
    raw_topic0: str | None
    topic_count: int
    status: str      # ok | zero_topic

    @property
    def token(self) -> str | None:
        return f"{NS_TOPIC0}:{self.raw_topic0}" if self.raw_topic0 else None


def receipt_topics(tx: NormalizedTx) -> list[TopicObservation] | None:
    """Observations of each receipt log's first topic, in log order.

    Returns None when no receipt was read at all - which is different from a
    receipt with an empty log list, a valid and common outcome that yields an
    empty list and therefore genuine zero counts.
    """
    if not tx.logs_available:
        return None
    out: list[TopicObservation] = []
    for log in tx.logs:
        out.append(TopicObservation(
            log_index=log.log_index, address=log.address, raw_topic0=log.raw_topic0,
            topic_count=log.topic_count,
            status="zero_topic" if log.is_zero_topic else "ok"))
    return out


# --- sparse token counts ------------------------------------------------------

def token_counts(tokens: Iterable[str | None]) -> dict[str, int]:
    """Sparse counts keyed by token, before any vocabulary is fitted."""
    return dict(Counter(t for t in tokens if t))


def tx_receipt_tokens(tx: NormalizedTx) -> tuple[dict[str, int], dict[str, object]]:
    """View 1: top-level selector candidate + receipt-log first-topic counts."""
    sel = top_level_selector(tx)
    topics = receipt_topics(tx)
    counts: dict[str, int] = {}
    if sel.token:
        counts[sel.token] = counts.get(sel.token, 0) + 1
    if topics is not None:
        for key, value in token_counts(t.token for t in topics).items():
            counts[key] = counts.get(key, 0) + value
    status = {
        "view": "tx-receipt",
        "top_selector_status": sel.status,
        "top_selector": sel.selector,
        "top_selector_call_type": sel.call_type,
        "receipt_available": topics is not None,
        "log_count": None if topics is None else len(topics),
        "zero_topic_logs": None if topics is None else sum(t.status == "zero_topic" for t in topics),
        "usable": topics is not None,
        "unusable_reason": None if topics is not None else "no receipt read",
    }
    return counts, status


def full_calls_receipt_tokens(tx: NormalizedTx) -> tuple[dict[str, int], dict[str, object]]:
    """View 2: selector candidates over every frame + the same topic counts.

    Usable only with a call tree the source asserts is complete *and* whose
    frames carry input bytes. Anything less stays missing rather than being
    filled with zeros.
    """
    selectors = all_frame_selectors(tx)
    topics = receipt_topics(tx)
    counts: dict[str, int] = {}
    for sel in selectors:
        if sel.token:
            counts[sel.token] = counts.get(sel.token, 0) + 1
    if topics is not None:
        for key, value in token_counts(t.token for t in topics).items():
            counts[key] = counts.get(key, 0) + value

    missing_input = sum(1 for s in selectors if s.status == SEL_UNAVAILABLE)
    tree_complete = tx.call_tree_status == TREE_COMPLETE
    usable = tree_complete and missing_input == 0 and topics is not None
    reasons = []
    if not tree_complete:
        reasons.append(f"call tree status is {tx.call_tree_status!r}, not complete")
    if missing_input:
        reasons.append(f"{missing_input} frame(s) supplied no input bytes")
    if topics is None:
        reasons.append("no receipt read")
    status = {
        "view": "full-calls-receipt",
        "frames": len(selectors),
        "root_frames": sum(1 for s in selectors if s.is_root),
        "call_tree_status": tx.call_tree_status,
        "call_tree_includes_root_duplicate": tx.call_tree_includes_root,
        "frames_without_input": missing_input,
        "reverted_frames": sum(1 for s in selectors if s.reverted),
        "creation_frames": sum(1 for s in selectors if s.status == SEL_CONTRACT_CREATION),
        "precompile_frames": sum(1 for s in selectors if s.status == SEL_PRECOMPILE),
        "receipt_available": topics is not None,
        "log_count": None if topics is None else len(topics),
        "usable": usable,
        "unusable_reason": "; ".join(reasons) or None,
    }
    return counts, status


def selector_status_summary(tx: NormalizedTx) -> dict[str, int]:
    return dict(Counter(s.status for s in all_frame_selectors(tx)))
