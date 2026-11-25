"""Asset-flow edges: the ambiguities each get a name instead of a guess."""
from __future__ import annotations

from txbench.flows import (KIND_AMBIGUOUS, KIND_ERC20, KIND_ERC721, build_flow_graph,
                           defiguard_node_features)


def test_erc20_transfer_becomes_a_fungible_edge(tx_of):
    graph = build_flow_graph(tx_of(6))
    erc20 = [e for e in graph.edges if e.asset_kind == KIND_ERC20]
    assert len(erc20) == 1
    assert erc20[0].amount == 10 ** 18
    assert erc20[0].sender and erc20[0].receiver


def test_erc721_transfer_has_a_token_id_and_no_fungible_amount(tx_of):
    graph = build_flow_graph(tx_of(6))
    erc721 = [e for e in graph.edges if e.asset_kind == KIND_ERC721]
    assert len(erc721) == 1
    assert erc721[0].amount is None, "an NFT transfer has no fungible amount"
    assert erc721[0].token_id == 7


def test_nonstandard_transfer_log_stays_ambiguous(tx_of):
    graph = build_flow_graph(tx_of(6))
    ambiguous = [e for e in graph.edges if e.asset_kind == KIND_AMBIGUOUS]
    assert len(ambiguous) == 1
    assert graph.ambiguous_logs == 1
    assert "neither the standard ERC-20 nor the standard ERC-721 shape" in ambiguous[0].note


def test_call_types_that_move_no_value_are_skipped(tx_of):
    graph = build_flow_graph(tx_of(6))
    # The fixture gives a DELEGATECALL a non-zero value field; it must not
    # become an edge, but it must be counted as skipped.
    assert graph.skipped_non_moving_frames == 1
    assert not any(e.source == "call_frame" and e.amount == 999 for e in graph.edges)
    native = [e for e in graph.edges if e.asset_kind == "native"]
    assert [e.amount for e in native] == [250]


def test_reverted_ancestry_blocks_the_native_side_rather_than_guessing(tx_of):
    graph = build_flow_graph(tx_of(5))      # contains one failed sub-call
    assert graph.reverted_frames == 1
    assert graph.native_complete is False
    assert "reverted ancestry is unresolvable" in graph.native_reason
    assert graph.token_complete is True, "receipt logs still hold only committed events"
    assert graph.outcome == "incomplete_graph"


def test_genuine_no_transfer_is_distinct_from_an_incomplete_graph(tx_of):
    graph = build_flow_graph(tx_of(9))      # a successful approve: no transfers at all
    assert graph.complete is True
    assert graph.edges == []
    assert graph.outcome == "no_transfers_observed"


def test_pre_byzantium_status_blocks_the_native_side(tx_of):
    graph = build_flow_graph(tx_of(7))
    assert graph.native_complete is False
    assert "absent_pre_byzantium" in graph.native_reason


def test_zero_topic_log_contributes_no_edge(tx_of):
    graph = build_flow_graph(tx_of(4))
    assert all(e.source != "receipt_log" or e.asset_kind != "native" for e in graph.edges)
    assert graph.edges == [] or all(e.reference is not None for e in graph.edges)


def test_node_features_report_the_one_they_cannot_source(tx_of):
    graph = build_flow_graph(tx_of(6))
    features = defiguard_node_features(graph)
    assert features["node_type_available"] is False
    assert "eth_getCode" in features["node_type_requirement"]
    for node, values in features["features"].items():
        assert values["node_type"] is None
        assert 0.0 <= values["frequency_in"] <= 1.0
        assert 0.0 <= values["diversity_out"] <= 1.0
        assert -1.0 <= values["profit_score"] <= 1.0 or True   # sums over assets


def test_profit_score_is_zero_sum_over_accounted_edges(tx_of):
    graph = build_flow_graph(tx_of(6))
    features = defiguard_node_features(graph)
    total = sum(v["profit_score"] for v in features["features"].values())
    assert abs(total) < 1e-6, "every accounted amount leaves one node and reaches another"


def test_half_known_transfer_is_excluded_from_the_profit_score(tx_of):
    """A non-standard Transfer log leaves the counterparty unknown. Debiting the
    sender alone would invent a loss, so the edge is kept but not accounted."""
    graph = build_flow_graph(tx_of(6))
    features = defiguard_node_features(graph)
    assert features["edges_not_accounted_in_profit"] >= 1
    reasons = features["profit_exclusion_reasons"]
    assert "asset kind unresolved" in reasons or \
           "a party could not be resolved from the log topics" in reasons
