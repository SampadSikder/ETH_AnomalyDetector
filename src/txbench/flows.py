"""Directed asset-flow edges: native value moves and token transfer logs.

This is the input DeFiGuard's cash-flow graph is built from, and it is where
the ambiguities live, so each one gets a name instead of a guess:

  * **call types that move no value** - DELEGATECALL and STATICCALL cannot
    transfer the callee's balance, so their `value` field never becomes an edge;
  * **reverted ancestry** - a frame undone by an *ancestor's* failure leaves no
    state change, but a flat frame list with no parent pointers cannot say who
    an ancestor is. Native edges are therefore emitted only from a transaction
    whose receipt succeeded and whose frames report no error; otherwise the
    native side is marked incomplete rather than guessed;
  * **ERC-20 / ERC-721 signature ambiguity** - `Transfer(address,address,uint256)`
    and ERC-721's `Transfer(address,address,uint256)` share one topic 0. The
    indexing differs (a fungible transfer carries the amount in `data`; an NFT
    transfer indexes the token id as a third topic), which separates the
    standard cases but not the non-standard ones, and those stay `ambiguous`;
  * a transaction with **no transfers at all** is a real outcome and is
    reported separately from an **incomplete graph**.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .trace_model import NormalizedTx

#: keccak256("Transfer(address,address,uint256)") - shared by ERC-20 and ERC-721.
TRANSFER_TOPIC0 = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
#: keccak256("TransferSingle(address,address,address,uint256,uint256)") - ERC-1155.
ERC1155_SINGLE_TOPIC0 = "0xc3d58168c5ae7397731d063d5bbf3d657854427343f4c083240f7aacaa2d0f62"
#: keccak256("TransferBatch(address,address,address,uint256[],uint256[])") - ERC-1155.
ERC1155_BATCH_TOPIC0 = "0x4a39dc06d4c0dbc64b70af90fd698a233a518aa5d07e595d983b8c0526c8f7fb"

NATIVE_ASSET = "native"

#: Frame types that can move the caller's native balance.
VALUE_MOVING_TYPES = frozenset({"CALL", "CALLCODE", "CREATE", "CREATE2"})
#: Frame types that carry a value field that is *not* a transfer.
NON_MOVING_TYPES = frozenset({"DELEGATECALL", "STATICCALL"})

KIND_ERC20 = "erc20"
KIND_ERC721 = "erc721"
KIND_ERC1155 = "erc1155"
KIND_AMBIGUOUS = "ambiguous"


@dataclass
class FlowEdge:
    sender: str | None
    receiver: str | None
    asset: str                 # "native" or a token contract address
    asset_kind: str            # native | erc20 | erc721 | erc1155 | ambiguous
    amount: int | None         # None when the standard gives no fungible amount
    source: str                # call_frame | receipt_log
    reference: int | None      # frame index or log index
    token_id: int | None = None
    note: str | None = None

    def as_row(self) -> dict[str, Any]:
        return {"sender": self.sender, "receiver": self.receiver, "asset": self.asset,
                "asset_kind": self.asset_kind, "amount": None if self.amount is None else str(self.amount),
                "source": self.source, "reference": self.reference, "token_id":
                None if self.token_id is None else str(self.token_id), "note": self.note}


@dataclass
class FlowGraph:
    tx_key: str
    edges: list[FlowEdge] = field(default_factory=list)
    native_complete: bool = False
    native_reason: str | None = None
    token_complete: bool = False
    token_reason: str | None = None
    ambiguous_logs: int = 0
    skipped_non_moving_frames: int = 0
    reverted_frames: int = 0

    @property
    def complete(self) -> bool:
        return self.native_complete and self.token_complete

    @property
    def outcome(self) -> str:
        """`no_transfers_observed` is a genuine result; `incomplete_graph` is not."""
        if not self.complete:
            return "incomplete_graph"
        return "transfers_observed" if self.edges else "no_transfers_observed"

    def nodes(self) -> list[str]:
        seen = {e.sender for e in self.edges} | {e.receiver for e in self.edges}
        return sorted(a for a in seen if a)

    def summary(self) -> dict[str, Any]:
        return {
            "tx_key": self.tx_key, "outcome": self.outcome,
            "edges": len(self.edges), "nodes": len(self.nodes()),
            "native_edges": sum(1 for e in self.edges if e.asset_kind == "native"),
            "token_edges": sum(1 for e in self.edges if e.asset_kind != "native"),
            "ambiguous_edges": sum(1 for e in self.edges if e.asset_kind == KIND_AMBIGUOUS),
            "ambiguous_logs": self.ambiguous_logs,
            "native_complete": self.native_complete, "native_reason": self.native_reason,
            "token_complete": self.token_complete, "token_reason": self.token_reason,
            "skipped_non_moving_frames": self.skipped_non_moving_frames,
            "reverted_frames": self.reverted_frames,
            "distinct_assets": len({e.asset for e in self.edges}),
        }


def _topic_address(topic: str | None) -> str | None:
    if not topic or len(topic) < 42:
        return None
    return "0x" + topic[-40:].lower()


def _hex_int(value: str | None) -> int | None:
    if not value:
        return None
    body = value[2:] if value.startswith("0x") else value
    if not body:
        return None
    try:
        return int(body, 16)
    except ValueError:
        return None


def build_flow_graph(tx: NormalizedTx) -> FlowGraph:
    graph = FlowGraph(tx_key=tx.tx_key)

    # --- native value, from the call frames -----------------------------------
    frames_errored = [f for f in tx.frames if f.error]
    graph.reverted_frames = len(frames_errored)
    succeeded = tx.receipt_status == 1
    if tx.call_tree_status != "complete":
        graph.native_reason = (f"call tree status is {tx.call_tree_status!r}: native value "
                               "moves cannot be enumerated")
    elif not succeeded and tx.receipt_status is not None:
        graph.native_reason = ("the transaction reverted: its native moves were undone, and a "
                               "flat frame list cannot show which frames committed")
    elif tx.receipt_status is None:
        graph.native_reason = (f"receipt status unavailable ({tx.receipt_status_source}): "
                               "cannot tell which frames committed")
    elif frames_errored and not tx.call_tree_has_nesting:
        graph.native_reason = (f"{len(frames_errored)} frame(s) failed and the source supplies "
                               "no parent pointers, so reverted ancestry is unresolvable")
    else:
        graph.native_complete = True
        for frame in tx.frames:
            call_type = (frame.call_type or "").upper()
            if call_type in NON_MOVING_TYPES:
                if frame.value_wei:
                    graph.skipped_non_moving_frames += 1
                continue
            if call_type not in VALUE_MOVING_TYPES and call_type != "UNKNOWN":
                continue
            if not frame.value_wei:
                continue
            graph.edges.append(FlowEdge(
                sender=frame.from_address, receiver=frame.to_address, asset=NATIVE_ASSET,
                asset_kind="native", amount=int(frame.value_wei), source="call_frame",
                reference=frame.index,
                note="outer call" if frame.is_root else None))

    # --- token transfers, from the receipt logs -------------------------------
    if not tx.logs_available:
        graph.token_reason = "no receipt read: token transfers cannot be enumerated"
        return graph
    graph.token_complete = True
    for log in tx.logs:
        topic0 = log.raw_topic0
        if topic0 is None:
            continue                               # a zero-topic log carries no signature
        if topic0 == TRANSFER_TOPIC0:
            graph.edges.append(_transfer_edge(log, graph))
        elif topic0 == ERC1155_SINGLE_TOPIC0:
            graph.edges.append(FlowEdge(
                sender=_topic_address(log.topics[2] if len(log.topics) > 2 else None),
                receiver=_topic_address(log.topics[3] if len(log.topics) > 3 else None),
                asset=log.address or "unknown", asset_kind=KIND_ERC1155, amount=None,
                source="receipt_log", reference=log.log_index,
                note="ERC-1155 TransferSingle: id and value live in data; not decoded here"))
        elif topic0 == ERC1155_BATCH_TOPIC0:
            graph.ambiguous_logs += 1
            graph.edges.append(FlowEdge(
                sender=_topic_address(log.topics[2] if len(log.topics) > 2 else None),
                receiver=_topic_address(log.topics[3] if len(log.topics) > 3 else None),
                asset=log.address or "unknown", asset_kind=KIND_AMBIGUOUS, amount=None,
                source="receipt_log", reference=log.log_index,
                note="ERC-1155 TransferBatch: dynamic arrays in data; amounts not decoded here"))
    return graph


def _transfer_edge(log, graph: FlowGraph) -> FlowEdge:
    """Separate a fungible Transfer from an NFT Transfer where the standards allow it."""
    topics = log.topics
    data_bytes = 0 if log.data_hex in (None, "0x") else (len(log.data_hex) - 2) // 2
    sender = _topic_address(topics[1]) if len(topics) > 1 else None
    receiver = _topic_address(topics[2]) if len(topics) > 2 else None

    if len(topics) == 4 and data_bytes == 0:
        return FlowEdge(sender=sender, receiver=receiver, asset=log.address or "unknown",
                        asset_kind=KIND_ERC721, amount=None,
                        token_id=_hex_int(topics[3]), source="receipt_log",
                        reference=log.log_index,
                        note="three indexed parameters and empty data: an ERC-721 transfer "
                             "of one token id, which has no fungible amount")
    if len(topics) == 3 and data_bytes == 32:
        return FlowEdge(sender=sender, receiver=receiver, asset=log.address or "unknown",
                        asset_kind=KIND_ERC20, amount=_hex_int(log.data_hex),
                        source="receipt_log", reference=log.log_index)
    graph.ambiguous_logs += 1
    return FlowEdge(sender=sender, receiver=receiver, asset=log.address or "unknown",
                    asset_kind=KIND_AMBIGUOUS, amount=_hex_int(log.data_hex) if data_bytes == 32 else None,
                    source="receipt_log", reference=log.log_index,
                    note=f"Transfer topic 0 with {len(topics)} topic(s) and {data_bytes} data "
                         "byte(s): neither the standard ERC-20 nor the standard ERC-721 shape, "
                         "so the asset kind stays unresolved")


def defiguard_node_features(graph: FlowGraph, *,
                            account_types: dict[str, str] | None = None) -> dict[str, Any]:
    """The paper's four node features, with the one we cannot source named.

    From the paper (Table I): node type (account type *and* source-code
    transparency), transfer frequency, transfer diversity, profit score.
    Three are computable from the edges alone. Node type is not: it needs each
    address's account kind (an `eth_getCode` read) and its verification status
    (an external explorer), so it is reported as missing rather than defaulted.
    """
    nodes = graph.nodes()
    in_count = {n: 0 for n in nodes}
    out_count = {n: 0 for n in nodes}
    in_assets: dict[str, set[str]] = {n: set() for n in nodes}
    out_assets: dict[str, set[str]] = {n: set() for n in nodes}
    profit: dict[str, float] = {n: 0.0 for n in nodes}
    max_amount: dict[str, int] = {}
    priced = 0
    unaccounted: dict[str, int] = {}

    def accountable(edge: FlowEdge) -> str | None:
        """Why an edge may not move profit. None means it may."""
        if edge.amount is None:
            return "no fungible amount in this standard"
        if edge.asset_kind == KIND_AMBIGUOUS:
            return "asset kind unresolved"
        if edge.sender is None or edge.receiver is None:
            return "a party could not be resolved from the log topics"
        return None

    for edge in graph.edges:
        if accountable(edge) is None:
            max_amount[edge.asset] = max(max_amount.get(edge.asset, 0), int(edge.amount))
    for edge in graph.edges:
        if edge.sender in out_count:
            out_count[edge.sender] += 1
            out_assets[edge.sender].add(edge.asset)
        if edge.receiver in in_count:
            in_count[edge.receiver] += 1
            in_assets[edge.receiver].add(edge.asset)
        # The profit score is a balance, so a half-known transfer must not
        # enter it: debiting a sender whose counterparty is unknown would
        # invent a loss. Such edges are counted and reported instead.
        reason = accountable(edge)
        if reason is not None:
            unaccounted[reason] = unaccounted.get(reason, 0) + 1
            continue
        scale = max_amount.get(edge.asset) or 1
        share = int(edge.amount) / scale
        priced += 1
        profit[edge.sender] -= share
        profit[edge.receiver] += share

    max_in = max(in_count.values(), default=0) or 1
    max_out = max(out_count.values(), default=0) or 1
    max_in_assets = max((len(v) for v in in_assets.values()), default=0) or 1
    max_out_assets = max((len(v) for v in out_assets.values()), default=0) or 1
    features = {}
    for node in nodes:
        features[node] = {
            "node_type": (account_types or {}).get(node),
            "frequency_in": round(in_count[node] / max_in, 6),
            "frequency_out": round(out_count[node] / max_out, 6),
            "diversity_in": round(len(in_assets[node]) / max_in_assets, 6),
            "diversity_out": round(len(out_assets[node]) / max_out_assets, 6),
            "profit_score": round(profit[node], 6),
        }
    return {
        "nodes": len(nodes),
        "features": features,
        "edges_accounted_in_profit": priced,
        "edges_not_accounted_in_profit": len(graph.edges) - priced,
        "profit_exclusion_reasons": unaccounted,
        "node_type_available": bool(account_types),
        "node_type_requirement": "per the paper, account kind (EOA or contract) *and* "
                                 "source-code transparency. Account kind needs one "
                                 "eth_getCode per distinct address; transparency needs an "
                                 "external source-verification index, which this project "
                                 "does not have.",
        "normalisation": "frequency and diversity are divided by the graph's maximum, and the "
                         "profit score by the per-asset maximum amount, as in the paper",
    }
