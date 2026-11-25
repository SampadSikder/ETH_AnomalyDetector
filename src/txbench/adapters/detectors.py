"""The five detectors: pinned specs and per-transaction checks."""
from __future__ import annotations

from typing import Any

from ..extract import full_calls_receipt_tokens, tx_receipt_tokens
from ..flows import build_flow_graph, defiguard_node_features
from ..trace_model import (NormalizedTx, RECEIPT_OK, RECEIPT_PRE_BYZANTIUM, TREE_COMPLETE)
from .base import (BLOCKED, DetectorSpec, FieldRequirement as F, PAPER_REIMPL, PARTIAL,
                   PROJECT_IMPL, SCHEMA_CHECK, TxCheck, UPSTREAM_DEGRADED, USABLE,
                   blocked_for_missing_record)

# --------------------------------------------------------------------------- #
# FT-Transformer (this project's supervised baseline)
# --------------------------------------------------------------------------- #
FT_TRANSFORMER = DetectorSpec(
    key="ft_transformer",
    name="FT-Transformer (supervised tabular baseline)",
    paper="Revisiting Deep Learning Models for Tabular Data (FT-Transformer)",
    paper_id="arXiv:2106.11959",
    code_repo=None, code_revision=None,
    preprocess_entry="txbench.features.build_sparse_view",
    sample_schema="data/features/<view>/feature_dictionary.json",
    requirements=(
        F("receipt", "a receipt, so that log topics can be counted; an empty log list is valid",
          source="upstream enriched trace or eth_getTransactionReceipt"),
        F("top_level_input", "the transaction's own calldata, for the top-level selector candidate",
          source="upstream enriched trace or eth_getTransactionByHash"),
        F("complete_call_tree", "a call tree asserted complete, with input bytes on every frame; "
          "needed for the full-calls view only", required=False,
          source="upstream enriched trace (release trace_status) or a call-tracer export"),
        F("label", "the cohort label, joined by (chain_id, tx_hash) for supervised training",
          source="the frozen manifest"),
    ),
    attainable_evidence=(SCHEMA_CHECK, PROJECT_IMPL),
    evidence_notes="Both documented views are exported by this project, so the preprocessing "
                   "entry point is ours; coverage is reported per view, separately.",
)


def check_ft_transformer(tx_key: str, tx: NormalizedTx | None) -> TxCheck:
    if tx is None:
        return blocked_for_missing_record(FT_TRANSFORMER.key, tx_key)
    _, view1 = tx_receipt_tokens(tx)
    _, view2 = full_calls_receipt_tokens(tx)
    available, missing, reasons = [], [], {}
    if view1["receipt_available"]:
        available.append("receipt")
    else:
        missing.append("receipt")
        reasons["receipt"] = "no receipt read for this transaction"
    if tx.input_hex is not None:
        available.append("top_level_input")
    else:
        missing.append("top_level_input")
        reasons["top_level_input"] = "the source supplied no top-level calldata"
    if view2["usable"]:
        available.append("complete_call_tree")
    else:
        missing.append("complete_call_tree")
        reasons["complete_call_tree"] = str(view2["unusable_reason"])
    verdict = USABLE if view1["usable"] and view2["usable"] else (
        PARTIAL if view1["usable"] else BLOCKED)
    return TxCheck(detector=FT_TRANSFORMER.key, tx_key=tx_key, verdict=verdict,
                   available_fields=available, missing_fields=missing, missing_reasons=reasons,
                   evidence=PROJECT_IMPL,
                   notes={"view_tx_receipt_usable": view1["usable"],
                          "view_full_calls_usable": view2["usable"],
                          "top_selector_status": view1["top_selector_status"],
                          "log_count": view1["log_count"]})


# --------------------------------------------------------------------------- #
# BlockScan
# --------------------------------------------------------------------------- #
BLOCKSCAN = DetectorSpec(
    key="blockscan",
    name="BlockScan",
    paper="BlockScan (authors' repository)",
    paper_id="github.com/nuwuxian/BlockScan",
    code_repo="https://github.com/nuwuxian/BlockScan",
    code_revision="4556cd209c968ed60e8d15b245f8d53e3f725534",
    preprocess_entry="src_eth/preprocess.py::extract_function_calls "
                     "(driver: write_all_traces)",
    sample_schema="{\"<tx_hash>\": [ {type, from, to, func, gas, value, "
                  "args:[{type,data}], output:[{type,data}], "
                  "calls:[<recursive>], logs:[{address, topics:[{type,data}], "
                  "data:[{type,data}]}] } ]}",
    requirements=(
        F("nested_calls", "a recursively nested `calls` array: the preprocessor walks children "
          "of each frame, so a flat frame list cannot be converted",
          source="a call-tracer export (debug_traceTransaction callTracer), a Parity-style "
                 "trace with trace_address, or the BigQuery traces table"),
        F("frame_decoded_args", "`args` / `output` as typed values ({type, data}) for every "
          "frame, not just the outer call", required=False,
          source="ABI decoding of each frame's input/output bytes against a verified ABI"),
        F("log_typed_topics", "each log's topics as typed values, with topic 0 present "
          "(`log['topics'][0]['data']` is indexed unconditionally, so a zero-topic log would "
          "raise)", source="receipt logs plus ABI decoding"),
        F("call_io_bytes", "input and output bytes for each frame",
          source="upstream enriched trace (input/output are supplied per frame)"),
    ),
    attainable_evidence=(SCHEMA_CHECK, UPSTREAM_DEGRADED),
    evidence_notes="The pinned revision contains no `config/` (top_20k_address.txt, "
                   "top_20k_hash.txt) and no `data/` example, so the full driver "
                   "`write_all_traces` cannot run as shipped; its two files are distributed "
                   "only through an external drive link. The trace builder "
                   "`extract_function_calls` is imported from the pinned file and run directly, "
                   "and because our converted record carries no nested `calls`, that run is "
                   "recorded as upstream_preprocess_degraded, never as a pass.",
)


def check_blockscan(tx_key: str, tx: NormalizedTx | None) -> TxCheck:
    if tx is None:
        return blocked_for_missing_record(BLOCKSCAN.key, tx_key)
    available, missing, reasons = [], [], {}
    if tx.call_tree_has_nesting:
        available.append("nested_calls")
    else:
        missing.append("nested_calls")
        reasons["nested_calls"] = (
            "the release supplies an ordered flat frame list with no trace address, parent "
            "pointer or depth, so the recursive `calls` array cannot be reconstructed")
    frames_with_input = sum(1 for f in tx.frames if f.input_hex is not None)
    if frames_with_input == len(tx.frames) and tx.frames:
        available.append("call_io_bytes")
    else:
        missing.append("call_io_bytes")
        reasons["call_io_bytes"] = (f"{len(tx.frames) - frames_with_input} of {len(tx.frames)} "
                                    "frames supplied no input bytes")
    decoded_frames = 0            # upstream decodes the outer call only
    if tx.logs_available:
        zero_topic = sum(1 for log in tx.logs if log.is_zero_topic)
        decoded_logs = sum(1 for log in tx.logs if log.decoded_signature)
        if zero_topic:
            missing.append("log_typed_topics")
            reasons["log_typed_topics"] = (
                f"{zero_topic} log(s) carry no topic 0; the preprocessor indexes topics[0] "
                "unconditionally and would raise")
        elif decoded_logs < len(tx.logs):
            missing.append("log_typed_topics")
            reasons["log_typed_topics"] = (
                f"only {decoded_logs} of {len(tx.logs)} logs carry a decoded, typed form "
                "upstream; the rest would be passed as raw bytes with no ABI types")
        else:
            available.append("log_typed_topics")
    else:
        missing.append("log_typed_topics")
        reasons["log_typed_topics"] = "no receipt read"
    missing.append("frame_decoded_args")
    reasons["frame_decoded_args"] = (
        "the release decodes the outer call only (`decoded_call`); internal frames carry raw "
        "input/output bytes with no ABI types")
    verdict = BLOCKED if "nested_calls" in missing else PARTIAL
    return TxCheck(detector=BLOCKSCAN.key, tx_key=tx_key, verdict=verdict,
                   available_fields=available, missing_fields=missing, missing_reasons=reasons,
                   evidence=SCHEMA_CHECK,
                   notes={"frames": len(tx.frames), "frames_with_input": frames_with_input,
                          "decoded_frames": decoded_frames,
                          "logs": tx.log_count})


# --------------------------------------------------------------------------- #
# BlockGPT
# --------------------------------------------------------------------------- #
BLOCKGPT = DetectorSpec(
    key="blockgpt",
    name="BlockGPT",
    paper="Blockchain Large Language Models",
    paper_id="arXiv:2304.12749",
    code_repo=None, code_revision=None,
    preprocess_entry=None,
    sample_schema="EVM trace tree encoding: a call tree with per-frame inputs and outputs, log "
                  "records, and the ordered storage reads and writes with their keys and values",
    requirements=(
        F("call_tree", "the call tree, with per-frame input and output bytes",
          source="upstream enriched trace (flat frames) plus nesting from a call tracer"),
        F("nested_structure", "the tree shape itself, for the paper's tree encoding",
          source="a call-tracer export or a Parity-style trace with trace_address"),
        F("log_records", "the receipt's log records",
          source="upstream enriched trace or eth_getTransactionReceipt"),
        F("ordered_state_access", "ordered storage reads and writes with keys and values. A "
          "pre/post state difference cannot reconstruct every intermediate read or write",
          source="debug_traceTransaction with a struct-log or prestate tracer on an archive "
                 "node; excluded from the free plans configured here"),
    ),
    attainable_evidence=(SCHEMA_CHECK,),
    evidence_notes="No authors' preprocessing entry point is pinned here, so any adapter would "
                   "be a paper-based reimplementation and is labelled as such. The ordered "
                   "state-access requirement is the binding gap.",
)


def check_blockgpt(tx_key: str, tx: NormalizedTx | None) -> TxCheck:
    if tx is None:
        return blocked_for_missing_record(BLOCKGPT.key, tx_key)
    available, missing, reasons = [], [], {}
    if tx.call_tree_status == TREE_COMPLETE:
        available.append("call_tree")
    else:
        missing.append("call_tree")
        reasons["call_tree"] = f"call tree status is {tx.call_tree_status!r}"
    if tx.call_tree_has_nesting:
        available.append("nested_structure")
    else:
        missing.append("nested_structure")
        reasons["nested_structure"] = (
            "frames are an ordered flat list; the tree encoding needs parent/child structure")
    if tx.logs_available:
        available.append("log_records")
    else:
        missing.append("log_records")
        reasons["log_records"] = "no receipt read"
    missing.append("ordered_state_access")
    reasons["ordered_state_access"] = (
        "no storage reads or writes are available: the release carries none, and the Debug and "
        "Trace APIs are excluded from the configured free plans")
    return TxCheck(detector=BLOCKGPT.key, tx_key=tx_key, verdict=BLOCKED,
                   available_fields=available, missing_fields=missing, missing_reasons=reasons,
                   evidence=SCHEMA_CHECK,
                   notes={"frames": len(tx.frames), "logs": tx.log_count,
                          "state_access_records": 0})


# --------------------------------------------------------------------------- #
# BlockLens
# --------------------------------------------------------------------------- #
BLOCKLENS = DetectorSpec(
    key="blocklens",
    name="BlockLens",
    paper="BlockLens (IACR ePrint 2025/1634, see Appendix A for the trace tokenizer)",
    paper_id="eprint.iacr.org/2025/1634",
    code_repo=None, code_revision=None,
    preprocess_entry=None,
    sample_schema="an ordered opcode sequence per transaction, with operands and the "
                  "stack / memory / storage information the trace tokenizer reads, with "
                  "transaction boundaries and a deterministic chunk order preserved",
    requirements=(
        F("opcode_sequence", "the ordered EVM steps of the transaction",
          source="debug_traceTransaction with struct logs on an archive node, or an existing "
                 "opcode-level trace export"),
        F("step_operands", "per-step operands and stack / memory / storage values",
          source="the same struct-log trace"),
        F("tx_boundaries", "transaction boundaries and a deterministic chunk order",
          source="derivable once an opcode sequence exists"),
    ),
    attainable_evidence=(SCHEMA_CHECK,),
    evidence_notes="Top-level calldata and a nested call tree cannot supply an instruction "
                   "sequence, so no amount of call-level enrichment unblocks this detector.",
)


def check_blocklens(tx_key: str, tx: NormalizedTx | None) -> TxCheck:
    if tx is None:
        return blocked_for_missing_record(BLOCKLENS.key, tx_key)
    reasons = {
        "opcode_sequence": "no opcode-level trace exists for this transaction in the release or "
                           "the cache, and the configured free plans exclude struct-log tracing",
        "step_operands": "operands and stack/memory/storage values come from the same "
                         "struct-log trace, which is unavailable",
    }
    return TxCheck(detector=BLOCKLENS.key, tx_key=tx_key, verdict=BLOCKED,
                   available_fields=["tx_boundaries"],
                   missing_fields=["opcode_sequence", "step_operands"],
                   missing_reasons=reasons, evidence=SCHEMA_CHECK,
                   notes={"opcode_steps": 0,
                          "note": "call-level data cannot substitute for an instruction stream"})


# --------------------------------------------------------------------------- #
# DeFiGuard
# --------------------------------------------------------------------------- #
DEFIGUARD = DetectorSpec(
    key="defiguard",
    name="DeFiGuard",
    paper="DeFiGuard: A Price Manipulation Detection Service in DeFi using Graph Neural Networks",
    paper_id="arXiv:2406.11157",
    code_repo=None, code_revision=None,
    preprocess_entry="txbench.flows.build_flow_graph + txbench.flows.defiguard_node_features "
                     "(paper-based reimplementation)",
    sample_schema="a cash-flow graph (V, E, X): edges carry asset address and amount; node "
                  "features are node type, transfer frequency, transfer diversity and profit "
                  "score (paper Table I)",
    requirements=(
        F("native_flows", "directed native-asset edges from call values, excluding call types "
          "that move no value and frames undone by a revert",
          source="upstream enriched trace frames (value per frame)"),
        F("token_flows", "directed ERC-20 edges from transfer logs, with parties, asset and "
          "amount", source="receipt logs"),
        F("node_type", "per the paper: account kind (EOA or contract) *and* source-code "
          "transparency", required=False,
          source="eth_getCode per distinct address for the kind; an external "
                 "source-verification index for transparency"),
        F("reverted_ancestry", "which frames were undone by an ancestor's failure",
          required=False,
          source="nesting from a call tracer; a flat frame list cannot express ancestry"),
    ),
    attainable_evidence=(SCHEMA_CHECK, PAPER_REIMPL),
    evidence_notes="The graph builder and three of the paper's four node features are "
                   "reimplemented here from the paper and are labelled as a reimplementation, "
                   "not as the authors' code. Node type is reported missing rather than "
                   "defaulted.",
)


def check_defiguard(tx_key: str, tx: NormalizedTx | None) -> TxCheck:
    if tx is None:
        return blocked_for_missing_record(DEFIGUARD.key, tx_key)
    graph = build_flow_graph(tx)
    features = defiguard_node_features(graph)
    available, missing, reasons = [], [], {}
    if graph.native_complete:
        available.append("native_flows")
    else:
        missing.append("native_flows")
        reasons["native_flows"] = graph.native_reason or "native edges could not be enumerated"
    if graph.token_complete:
        available.append("token_flows")
    else:
        missing.append("token_flows")
        reasons["token_flows"] = graph.token_reason or "token edges could not be enumerated"
    missing.append("node_type")
    reasons["node_type"] = ("account kind needs one eth_getCode per distinct address and "
                            "source-code transparency needs an external verification index; "
                            "neither is collected, so the feature is left missing")
    if tx.call_tree_has_nesting:
        available.append("reverted_ancestry")
    else:
        missing.append("reverted_ancestry")
        reasons["reverted_ancestry"] = ("a flat frame list cannot express which frames an "
                                        "ancestor's failure undid")
    verdict = PARTIAL if graph.complete else BLOCKED
    notes: dict[str, Any] = {**graph.summary(),
                             "node_feature_count_available": 3,
                             "node_feature_count_required": 4,
                             "nodes_with_features": features["nodes"],
                             "coverage_outcome": graph.outcome}
    return TxCheck(detector=DEFIGUARD.key, tx_key=tx_key, verdict=verdict,
                   available_fields=available, missing_fields=missing, missing_reasons=reasons,
                   evidence=PAPER_REIMPL, notes=notes)


SPECS: dict[str, DetectorSpec] = {
    FT_TRANSFORMER.key: FT_TRANSFORMER,
    BLOCKSCAN.key: BLOCKSCAN,
    BLOCKGPT.key: BLOCKGPT,
    BLOCKLENS.key: BLOCKLENS,
    DEFIGUARD.key: DEFIGUARD,
}

CHECKERS = {
    FT_TRANSFORMER.key: check_ft_transformer,
    BLOCKSCAN.key: check_blockscan,
    BLOCKGPT.key: check_blockgpt,
    BLOCKLENS.key: check_blocklens,
    DEFIGUARD.key: check_defiguard,
}
