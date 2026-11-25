"""Selector and topic extraction: every non-selector gets an explicit status."""
from __future__ import annotations

from conftest import H

from txbench import extract as X
from txbench.features import VIEW_FULL_CALLS, VIEW_TX_RECEIPT, build_sparse_view


def test_empty_calldata_is_empty_input_not_missing(tx_of):
    candidate = X.top_level_selector(tx_of(2))
    assert candidate.status == X.SEL_EMPTY_INPUT
    assert candidate.selector is None


def test_short_calldata_is_not_a_selector(tx_of):
    candidate = X.top_level_selector(tx_of(4))
    assert candidate.status == X.SEL_SHORT_INPUT
    assert candidate.selector is None


def test_contract_creation_has_its_own_status(tx_of):
    candidate = X.top_level_selector(tx_of(3))
    assert candidate.status == X.SEL_CONTRACT_CREATION


def test_selector_is_the_first_four_bytes(tx_of):
    candidate = X.top_level_selector(tx_of(1))
    assert candidate.status == X.SEL_OK
    assert candidate.selector == "0x12345678"
    assert candidate.token == "sel:0x12345678"


def test_precompile_bytes_never_enter_the_selector_namespace(tx_of):
    candidates = X.all_frame_selectors(tx_of(14))
    precompile = [c for c in candidates if c.status == X.SEL_PRECOMPILE]
    assert len(precompile) == 1
    assert precompile[0].token.startswith(f"{X.NS_PRECOMPILE}:")
    assert not any(c.token and c.token.startswith("sel:0xabababab") for c in candidates)


def test_outer_call_is_counted_exactly_once(tx_of):
    tx = tx_of(13)                      # its frame list repeats the outer call
    assert tx.call_tree_includes_root is True
    roots = [f for f in tx.frames if f.is_root]
    assert len(roots) == 1
    counts, status = X.full_calls_receipt_tokens(tx)
    assert status["root_frames"] == 1
    # the outer call's selector appears once, not twice
    assert counts["sel:0x12345678"] == 1


def test_zero_topic_log_is_preserved_and_claims_no_signature(tx_of):
    observations = X.receipt_topics(tx_of(4))
    assert observations is not None
    zero = [o for o in observations if o.status == "zero_topic"]
    assert len(zero) == 1
    assert zero[0].raw_topic0 is None
    assert zero[0].token is None, "a zero-topic log contributes no topic token"


def test_four_topic_log_keeps_its_first_topic_as_raw_topic0(tx_of):
    tx = tx_of(4)
    four = [l for l in tx.logs if l.topic_count == 4][0]
    assert four.raw_topic0 == "0x" + "ee" * 32
    assert four.decoded_signature is None, "no event signature is claimed from the bytes"


def test_empty_log_list_is_valid_and_yields_zeros_not_missing(tx_of):
    counts, status = X.tx_receipt_tokens(tx_of(2))
    assert status["receipt_available"] is True
    assert status["log_count"] == 0
    assert status["usable"] is True
    assert counts == {}, "no topics observed, and that is an observed absence"


def test_unavailable_trace_is_missing_not_zero(records):
    key = f"1:{H(8)}"
    assert records[key] is None, "no trace and no cache means no record at all"
    sparse = build_sparse_view(VIEW_TX_RECEIPT, [(key, None)])
    row = sparse.rows[0]
    assert row.status["usable"] is False
    assert row.counts == {}


def test_partial_call_tree_blocks_only_the_full_calls_view(tx_of):
    tx = tx_of(4)                       # marked 'partial' upstream
    assert tx.call_tree_status == "partial"
    _, view1 = X.tx_receipt_tokens(tx)
    _, view2 = X.full_calls_receipt_tokens(tx)
    assert view1["usable"] is True
    assert view2["usable"] is False
    assert "not complete" in view2["unusable_reason"]


def test_selectors_and_topics_use_separate_namespaces(tx_of):
    counts, _ = X.tx_receipt_tokens(tx_of(1))
    namespaces = {token.split(":", 1)[0] for token in counts}
    assert namespaces <= {X.NS_SELECTOR, X.NS_TOPIC0, X.NS_PRECOMPILE}
    assert any(n == X.NS_SELECTOR for n in namespaces)
    assert any(n == X.NS_TOPIC0 for n in namespaces)


def test_attempted_calls_counted_separately_from_committed_events(tx_of):
    sparse = build_sparse_view(VIEW_FULL_CALLS, [("1:" + H(5), tx_of(5))])
    status = sparse.rows[0].status
    assert status["attempted_calls"] == 3          # root + two sub-calls
    assert status["committed_receipt_events"] == 1  # one receipt log
    assert status["reverted_frames"] == 1
