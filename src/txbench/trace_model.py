"""The project's canonical, source-agnostic view of one transaction.

Built from an upstream enriched trace, from RPC responses, or from both. Every
field that a source failed to supply stays absent and is named in `field_gaps`:
an absent field is never rendered as a zero, a revert or a benign outcome.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ids import normalize_address, normalize_chain_id, normalize_hex_bytes, normalize_tx_hash

# --- statuses -----------------------------------------------------------------
RECEIPT_OK = "ok"                                   # a status field was supplied (0 or 1)
RECEIPT_PRE_BYZANTIUM = "absent_pre_byzantium"      # pre-EIP-658: root, no status
RECEIPT_UNAVAILABLE = "unavailable"                 # no receipt read at all

TREE_COMPLETE = "complete"        # the source asserts a full internal call tree
TREE_PARTIAL = "partial"          # the source asserts an incomplete tree
TREE_UNAVAILABLE = "unavailable"  # no call-tree data at all
TREE_UNKNOWN = "unknown"          # frames present, completeness not asserted

CALL_TYPES_WITH_INPUT = {"CALL", "DELEGATECALL", "STATICCALL", "CALLCODE"}
CREATE_TYPES = {"CREATE", "CREATE2"}

#: Frame kinds that cannot carry calldata, whatever the source. SELFDESTRUCT
#: takes no arguments and returns nothing: it sweeps the balance to a
#: beneficiary. `SUICIDE` is the warehouse's spelling of the same opcode, and
#: `REWARD` is a block-level trace, not a call. For these frames absent
#: input/output bytes are **not applicable**, not missing - no source could ever
#: supply them, so counting them as a gap would state a permanently unmeetable
#: requirement. A CREATE is deliberately not here: its init code is real
#: calldata, carried in `action.init`.
NO_CALLDATA_TYPES = {"SELFDESTRUCT", "SUICIDE", "REWARD"}


def frame_takes_calldata(frame: "CallFrame") -> bool:
    """False when the frame's kind makes call I/O bytes inapplicable."""
    return (frame.call_type or "UNKNOWN").upper() not in NO_CALLDATA_TYPES


@dataclass
class ReceiptLog:
    """One receipt log, kept in full and in order.

    `raw_topic0` is the observed first topic and nothing more. A non-anonymous
    ABI event ordinarily puts its signature hash there, but an anonymous event
    may use the slot for an argument and a zero-topic log has no first topic at
    all, so no event signature is claimed here.
    """
    log_index: int | None
    address: str | None
    topics: list[str] = field(default_factory=list)
    data_hex: str | None = None
    #: Upstream's own decoding, carried as provenance only - never as evidence.
    decoded_label: str | None = None
    decoded_signature: str | None = None

    @property
    def raw_topic0(self) -> str | None:
        return self.topics[0] if self.topics else None

    @property
    def topic_count(self) -> int:
        return len(self.topics)

    @property
    def is_zero_topic(self) -> bool:
        return not self.topics


@dataclass
class CallFrame:
    """One call frame. `parent_index`/`depth` are None when the source supplies
    no nesting information (a flat ordered list of frames)."""
    index: int
    call_type: str
    from_address: str | None = None
    to_address: str | None = None
    value_wei: int | None = None
    input_hex: str | None = None
    output_hex: str | None = None
    error: str | None = None
    gas: int | None = None
    gas_used: int | None = None
    parent_index: int | None = None
    depth: int | None = None
    is_root: bool = False
    #: True when this frame, or an ancestor, failed.
    reverted: bool | None = None


@dataclass
class NormalizedTx:
    chain_id: int
    tx_hash: str
    # envelope
    from_address: str | None = None
    to_address: str | None = None
    value_wei: int | None = None
    input_hex: str | None = None
    nonce: int | None = None
    gas: int | None = None
    gas_price: int | None = None
    tx_type: int | None = None
    is_contract_creation: bool | None = None
    # chain location
    block_number: int | None = None
    block_hash: str | None = None
    block_timestamp: str | None = None
    transaction_index: int | None = None
    # receipt
    receipt_status: int | None = None
    receipt_status_source: str = RECEIPT_UNAVAILABLE
    receipt_gas_used: int | None = None
    receipt_contract_address: str | None = None
    receipt_root: str | None = None
    logs_available: bool = False
    logs: list[ReceiptLog] = field(default_factory=list)
    # internal calls
    frames: list[CallFrame] = field(default_factory=list)
    call_tree_status: str = TREE_UNAVAILABLE
    call_tree_has_nesting: bool = False
    call_tree_includes_root: bool = False
    #: True once a source supplied real parent pointers, so each frame's
    #: `reverted` is either a decided bool or an explicit unknown (None).
    #: A flat frame list leaves this false: nothing was ever decided.
    call_tree_ancestry_known: bool = False
    # provenance
    sources: dict[str, str] = field(default_factory=dict)   # section -> source name
    field_gaps: dict[str, str] = field(default_factory=dict)  # field -> reason

    @property
    def tx_key(self) -> str:
        return f"{self.chain_id}:{self.tx_hash}"

    @property
    def log_count(self) -> int | None:
        return len(self.logs) if self.logs_available else None

    def note_gap(self, field_name: str, reason: str) -> None:
        self.field_gaps.setdefault(field_name, reason)


# --- parsing helpers ----------------------------------------------------------

def _int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(text, 10)
    except ValueError:
        return None


def _status_from_upstream(raw: Any, receipt_root: Any) -> tuple[int | None, str]:
    """An empty status with a receipt root is a pre-Byzantium receipt.

    It is NOT a revert and NOT a success: the field did not exist before
    EIP-658, and no RPC read can recover it.
    """
    if raw is None or str(raw).strip() == "":
        if receipt_root:
            return None, RECEIPT_PRE_BYZANTIUM
        return None, RECEIPT_UNAVAILABLE
    value = _int(raw)
    if value in (0, 1):
        return value, RECEIPT_OK
    return None, RECEIPT_UNAVAILABLE


def _topics_from_upstream(raw_log: dict[str, Any]) -> list[str]:
    """Rebuild ordered topics from `topic0..topic3`, stopping at the first gap.

    A log with no `topic0` is a zero-topic log and keeps an empty topic list.
    """
    topics: list[str] = []
    for i in range(4):
        value = raw_log.get(f"topic{i}")
        if value is None or str(value).strip() == "":
            break
        topics.append(str(value).strip().lower())
    if not topics and isinstance(raw_log.get("topics"), list):
        topics = [str(t).strip().lower() for t in raw_log["topics"] if t]
    return topics


def from_upstream_trace(raw: dict[str, Any], *, chain_id: int,
                        tx_hash: str | None = None,
                        trace_status: str | None = None,
                        source: str = "upstream_enriched_trace") -> NormalizedTx:
    """Normalise one upstream enriched trace record.

    `trace_status` is the release's own assessment of the trace
    (`ok`/`partial`/`error`/`not_fetched`); without it the call tree's
    completeness is recorded as `unknown` rather than assumed complete.
    """
    tx = NormalizedTx(
        chain_id=normalize_chain_id(chain_id),
        tx_hash=normalize_tx_hash(tx_hash if tx_hash is not None else raw.get("hash")),
    )
    tx.sources["envelope"] = source
    tx.from_address = normalize_address(raw.get("from_address"))
    tx.to_address = normalize_address(raw.get("to_address"))
    tx.value_wei = _int(raw.get("value"))
    tx.input_hex = normalize_hex_bytes(raw.get("input"))
    tx.nonce = _int(raw.get("nonce"))
    tx.gas = _int(raw.get("gas"))
    tx.gas_price = _int(raw.get("gas_price"))
    tx.block_number = _int(raw.get("block_number"))
    tx.block_hash = (str(raw["block_hash"]).strip().lower() or None) if raw.get("block_hash") else None
    tx.block_timestamp = raw.get("block_timestamp") or None
    tx.transaction_index = _int(raw.get("transaction_index"))

    receipt_contract = normalize_address(raw.get("receipt_contract_address"))
    tx.receipt_contract_address = receipt_contract
    tx.is_contract_creation = bool(receipt_contract) or tx.to_address is None
    tx.receipt_root = (str(raw["receipt_root"]).strip().lower() or None) if raw.get("receipt_root") else None
    tx.receipt_gas_used = _int(raw.get("receipt_gas_used"))
    tx.receipt_status, tx.receipt_status_source = _status_from_upstream(
        raw.get("receipt_status"), tx.receipt_root)
    if tx.receipt_status_source == RECEIPT_PRE_BYZANTIUM:
        tx.note_gap("receipt_status", "pre-EIP-658 receipt: no status field exists on chain")
    elif tx.receipt_status_source == RECEIPT_UNAVAILABLE:
        tx.note_gap("receipt_status", "no receipt status supplied by the source")

    raw_logs = raw.get("logs")
    if isinstance(raw_logs, list):
        tx.logs_available = True
        tx.sources["receipt_logs"] = source
        for raw_log in raw_logs:
            decoded = raw_log.get("decoded_event") or {}
            tx.logs.append(ReceiptLog(
                log_index=_int(raw_log.get("log_index")),
                address=normalize_address(raw_log.get("address")),
                topics=_topics_from_upstream(raw_log),
                data_hex=normalize_hex_bytes(raw_log.get("data")),
                decoded_label=(decoded.get("label") if isinstance(decoded, dict) else None),
                decoded_signature=(decoded.get("signature") if isinstance(decoded, dict) else None),
            ))
        tx.logs.sort(key=lambda l: (l.log_index if l.log_index is not None else 1 << 40))
    else:
        tx.note_gap("receipt_logs", "the source supplied no log list")

    frames: list[CallFrame] = []
    root_input = tx.input_hex
    root_type = "CREATE" if tx.is_contract_creation else "CALL"
    frames.append(CallFrame(
        index=0, call_type=root_type, from_address=tx.from_address, to_address=tx.to_address,
        value_wei=tx.value_wei, input_hex=root_input, gas=tx.gas, gas_used=tx.receipt_gas_used,
        parent_index=None, depth=0, is_root=True,
        reverted=(tx.receipt_status == 0) if tx.receipt_status is not None else None,
    ))

    raw_internal = raw.get("internal_transactions")
    if isinstance(raw_internal, list):
        tx.sources["call_tree"] = source
        for i, raw_frame in enumerate(raw_internal, start=1):
            call_type = str(raw_frame.get("type") or "").strip().upper() or "UNKNOWN"
            frame = CallFrame(
                index=i,
                call_type=call_type,
                from_address=normalize_address(raw_frame.get("from")),
                to_address=normalize_address(raw_frame.get("to")),
                value_wei=_int(raw_frame.get("value")),
                input_hex=normalize_hex_bytes(raw_frame.get("input")),
                output_hex=normalize_hex_bytes(raw_frame.get("output")),
                error=(str(raw_frame.get("error")) if raw_frame.get("error") else None),
                gas=_int(raw_frame.get("gas")),
                gas_used=_int(raw_frame.get("gas_used")),
                parent_index=_int(raw_frame.get("parent_index")),
                depth=_int(raw_frame.get("depth")),
            )
            frame.reverted = True if frame.error else None
            frames.append(frame)
        # The upstream format lists sub-calls only; guard anyway so a source
        # that does include the root can never double-count it.
        tx.call_tree_includes_root = _detects_duplicated_root(frames)
        if tx.call_tree_includes_root:
            frames = [frames[0]] + [f for f in frames[1:] if not _same_call(f, frames[0])]
            for new_index, frame in enumerate(frames):
                frame.index = new_index
        tx.call_tree_has_nesting = any(f.parent_index is not None or f.depth is not None
                                       for f in frames[1:])
        if not tx.call_tree_has_nesting and len(frames) > 1:
            tx.note_gap("call_tree_nesting",
                        "the source supplies an ordered flat frame list with no "
                        "trace address, parent pointer or depth")
        status = (trace_status or "").strip().lower()
        if status == "ok":
            tx.call_tree_status = TREE_COMPLETE
        elif status in {"partial", "error"}:
            tx.call_tree_status = TREE_PARTIAL
            tx.note_gap("call_tree", f"the release marks this trace {status!r}")
        elif status == "not_fetched":
            tx.call_tree_status = TREE_UNAVAILABLE
            tx.note_gap("call_tree", "the release marks this trace 'not_fetched'")
        else:
            tx.call_tree_status = TREE_UNKNOWN
            tx.note_gap("call_tree_completeness",
                        "no trace_status supplied; completeness not asserted")
    else:
        tx.call_tree_status = TREE_UNAVAILABLE
        tx.note_gap("call_tree", "the source supplied no internal call list")

    tx.frames = frames
    return tx


def _same_call(a: CallFrame, b: CallFrame) -> bool:
    return (a.call_type == b.call_type and a.from_address == b.from_address
            and a.to_address == b.to_address and a.input_hex == b.input_hex)


def _detects_duplicated_root(frames: list[CallFrame]) -> bool:
    return any(_same_call(f, frames[0]) for f in frames[1:])


# --- RPC sources --------------------------------------------------------------

def from_rpc(tx_payload: dict[str, Any] | None, receipt_payload: dict[str, Any] | None, *,
             chain_id: int, tx_hash: str, block_payload: dict[str, Any] | None = None,
             source: str = "rpc") -> NormalizedTx:
    """Normalise `eth_getTransactionByHash` + `eth_getTransactionReceipt` replies.

    No call tree and no state access come from these two methods, so the call
    tree is recorded as unavailable rather than as an empty one.
    """
    tx = NormalizedTx(chain_id=normalize_chain_id(chain_id),
                      tx_hash=normalize_tx_hash(tx_hash))
    if tx_payload:
        tx.sources["envelope"] = source
        tx.from_address = normalize_address(tx_payload.get("from"))
        tx.to_address = normalize_address(tx_payload.get("to"))
        tx.value_wei = _int(tx_payload.get("value"))
        tx.input_hex = normalize_hex_bytes(tx_payload.get("input"))
        tx.nonce = _int(tx_payload.get("nonce"))
        tx.gas = _int(tx_payload.get("gas"))
        tx.gas_price = _int(tx_payload.get("gasPrice"))
        tx.tx_type = _int(tx_payload.get("type"))
        tx.block_number = _int(tx_payload.get("blockNumber"))
        tx.block_hash = _lower_or_none(tx_payload.get("blockHash"))
        tx.transaction_index = _int(tx_payload.get("transactionIndex"))
        tx.is_contract_creation = tx.to_address is None
    else:
        tx.note_gap("envelope", "no transaction envelope was read")

    if receipt_payload:
        tx.sources["receipt"] = source
        tx.receipt_gas_used = _int(receipt_payload.get("gasUsed"))
        tx.receipt_contract_address = normalize_address(receipt_payload.get("contractAddress"))
        tx.receipt_root = _lower_or_none(receipt_payload.get("root"))
        status = receipt_payload.get("status")
        if status is None:
            tx.receipt_status, tx.receipt_status_source = _status_from_upstream(
                None, tx.receipt_root)
            tx.note_gap("receipt_status",
                        "the receipt carries no status field"
                        + (": pre-EIP-658 receipt" if tx.receipt_root else ""))
        else:
            tx.receipt_status = _int(status)
            tx.receipt_status_source = (RECEIPT_OK if tx.receipt_status in (0, 1)
                                        else RECEIPT_UNAVAILABLE)
        if tx.block_number is None:
            tx.block_number = _int(receipt_payload.get("blockNumber"))
        if tx.block_hash is None:
            tx.block_hash = _lower_or_none(receipt_payload.get("blockHash"))
        if tx.transaction_index is None:
            tx.transaction_index = _int(receipt_payload.get("transactionIndex"))
        if tx.receipt_contract_address:
            tx.is_contract_creation = True
        raw_logs = receipt_payload.get("logs")
        if isinstance(raw_logs, list):
            tx.logs_available = True
            for raw_log in raw_logs:
                topics = [str(t).strip().lower() for t in (raw_log.get("topics") or []) if t]
                tx.logs.append(ReceiptLog(
                    log_index=_int(raw_log.get("logIndex")),
                    address=normalize_address(raw_log.get("address")),
                    topics=topics,
                    data_hex=normalize_hex_bytes(raw_log.get("data"))))
            tx.logs.sort(key=lambda l: (l.log_index if l.log_index is not None else 1 << 40))
        else:
            tx.note_gap("receipt_logs", "the receipt reply carried no log list")
    else:
        tx.note_gap("receipt", "no receipt was read")

    if block_payload:
        tx.sources["block"] = source
        tx.block_timestamp = _timestamp(block_payload.get("timestamp"))
        if tx.block_hash is None:
            tx.block_hash = _lower_or_none(block_payload.get("hash"))

    root = CallFrame(index=0, call_type="CREATE" if tx.is_contract_creation else "CALL",
                     from_address=tx.from_address, to_address=tx.to_address,
                     value_wei=tx.value_wei, input_hex=tx.input_hex, gas=tx.gas,
                     gas_used=tx.receipt_gas_used, depth=0, is_root=True,
                     reverted=(tx.receipt_status == 0) if tx.receipt_status is not None else None)
    tx.frames = [root]
    tx.call_tree_status = TREE_UNAVAILABLE
    tx.note_gap("call_tree", "eth_getTransactionByHash and eth_getTransactionReceipt do not "
                             "carry internal calls")
    return tx


def merge_rpc_into(tx: NormalizedTx, tx_payload: dict[str, Any] | None,
                   receipt_payload: dict[str, Any] | None,
                   block_payload: dict[str, Any] | None = None,
                   source: str = "rpc") -> NormalizedTx:
    """Fill only the gaps in an upstream-derived record. Never overwrites."""
    if receipt_payload:
        if tx.receipt_status is None and receipt_payload.get("status") is not None:
            value = _int(receipt_payload.get("status"))
            if value in (0, 1):
                tx.receipt_status = value
                tx.receipt_status_source = RECEIPT_OK
                tx.sources["receipt_status"] = source
                tx.field_gaps.pop("receipt_status", None)
        if not tx.logs_available and isinstance(receipt_payload.get("logs"), list):
            filled = from_rpc(None, receipt_payload, chain_id=tx.chain_id, tx_hash=tx.tx_hash)
            tx.logs = filled.logs
            tx.logs_available = True
            tx.sources["receipt_logs"] = source
            tx.field_gaps.pop("receipt_logs", None)
    if tx_payload:
        if tx.input_hex is None:
            tx.input_hex = normalize_hex_bytes(tx_payload.get("input"))
            tx.sources["input"] = source
        if tx.tx_type is None:
            tx.tx_type = _int(tx_payload.get("type"))
        if tx.transaction_index is None:
            tx.transaction_index = _int(tx_payload.get("transactionIndex"))
    if block_payload and tx.block_timestamp is None:
        tx.block_timestamp = _timestamp(block_payload.get("timestamp"))
        tx.sources["block"] = source
    return tx


def _lower_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _timestamp(value: Any) -> str | None:
    seconds = _int(value)
    if seconds is None:
        return None
    from datetime import datetime, timezone
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
