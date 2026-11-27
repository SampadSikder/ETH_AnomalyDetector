"""Call-tree reconstruction: the two cases that decide whether a flow is real.

A child can return successfully and still be undone, because its parent
reverted after it returned. And a tree with a hole in it cannot say whether
that happened at all. Both are tested here against the DeFiGuard rebuild, not
only against the tree, because the only thing that matters is whether a
transfer that never committed can reach the graph.
"""
from __future__ import annotations

import pytest

from txbench.calltree import (ISSUE_DUPLICATE_PATH, ISSUE_MISSING_CHILD, ISSUE_MISSING_PARENT,
                              ISSUE_NO_ROOT, TRACE_SOURCE_DEFAULT, CallTreeError, TraceRow,
                              apply_nesting, cross_check_root, parent_path, parse_trace_path,
                              reconstruct)
from txbench.flows import build_flow_graph
from txbench.trace_model import RECEIPT_OK, NormalizedTx

from conftest import ADDR, H

TX = H(0xAA)


def row(path, *, value=None, error=None, call_type="CALL", frm=1, to=2, subcalls=None,
        input_hex=None):
    return TraceRow(path=tuple(path), call_type=call_type, from_address=ADDR(frm),
                    to_address=ADDR(to), value_wei=value, error=error,
                    subcall_count=subcalls, input_hex=input_hex,
                    source=TRACE_SOURCE_DEFAULT)


def tx_with(frames_status=1):
    """A local record whose receipt committed, with no frames of its own yet."""
    return NormalizedTx(chain_id=1, tx_hash=TX, from_address=ADDR(1), to_address=ADDR(2),
                        value_wei=0, receipt_status=frames_status,
                        receipt_status_source=RECEIPT_OK, logs_available=True, logs=[],
                        call_tree_status="complete")


# --- path algebra -------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (None, ()), ("", ()), ("[]", ()), ("0", (0,)), ("0,1", (0, 1)), ("0_2_3", (0, 2, 3)),
    ([1, 0], (1, 0)), ((2,), (2,)), ("10,0", (10, 0)),
])
def test_trace_paths_parse_to_child_indexes(raw, expected):
    assert parse_trace_path(raw) == expected


@pytest.mark.parametrize("raw", ["a", "0,x", "-1", "0,-2"])
def test_an_unparsable_trace_path_is_an_error_not_a_dropped_frame(raw):
    with pytest.raises(CallTreeError):
        parse_trace_path(raw)


def test_the_parent_of_a_path_is_that_path_without_its_last_index():
    assert parent_path(()) is None
    assert parent_path((0,)) == ()
    assert parent_path((0, 1, 4)) == (0, 1)


# --- the required case 1: a successful child under a reverted parent ----------

def test_a_successful_child_under_a_reverted_parent_is_excluded_from_the_graph():
    """The child returned cleanly; its parent then reverted, undoing it.

    Row order and addresses give no hint of this - only the parent pointer
    does - so a flat frame list would have counted 500 wei as moved.
    """
    rows = [
        row([], value=0),                                   # root, succeeded
        row([0], value=100, frm=1, to=3),                   # sibling, committed
        row([1], value=700, frm=1, to=4, error="Reverted"),  # the parent that failed
        row([1, 0], value=500, frm=4, to=5),                # its child: succeeded locally
        row([1, 0, 0], value=50, frm=5, to=6),              # deeper still: also undone
    ]
    tree = reconstruct(TX, rows, source="test")
    assert tree.complete, tree.issues

    by_path = {p: tree.frames[i] for p, i in tree.index_by_path.items()}
    assert by_path[(0,)].reverted is False
    assert by_path[(1,)].reverted is True, "the frame that failed"
    assert by_path[(1, 0)].reverted is True, "a child undone by its parent's revert"
    assert by_path[(1, 0, 0)].reverted is True, "and its descendant"
    assert by_path[(1, 0)].error is None, "the child itself reported no error"

    tx = tx_with()
    apply_nesting(tx, tree, source="test")
    graph = build_flow_graph(tx)

    assert graph.native_complete, graph.native_reason
    assert graph.ancestry_resolved
    amounts = sorted(e.amount for e in graph.edges if e.asset_kind == "native")
    assert amounts == [100], "only the committed sibling moved value"
    assert graph.excluded_reverted_frames == 3, "the failed frame and its two descendants"


def test_the_same_frames_without_nesting_block_the_graph_instead():
    """The contrast that justifies the backfill: no parent pointers, no graph."""
    tx = tx_with()
    tx.frames = []
    tree = reconstruct(TX, [row([], value=0), row([1], value=700, error="Reverted"),
                            row([1, 0], value=500)], source="test")
    apply_nesting(tx, tree, source="test")
    tx.call_tree_has_nesting = False          # as the release supplies it
    tx.call_tree_ancestry_known = False       # nothing was ever decided
    for frame in tx.frames:
        frame.parent_index = None
        frame.reverted = None
    graph = build_flow_graph(tx)
    assert not graph.native_complete
    assert "no parent pointers" in graph.native_reason
    assert not any(e.asset_kind == "native" for e in graph.edges)


# --- the required case 2: a missing parent ------------------------------------

def test_a_missing_parent_leaves_the_revert_state_unknown_and_blocks_the_graph():
    """Path (2, 0) is present but (2,) is not, so nobody can say what undid it."""
    rows = [
        row([], value=0),
        row([0], value=100, frm=1, to=3),
        row([2, 0], value=400, frm=9, to=8),     # its parent (2,) never arrived
    ]
    tree = reconstruct(TX, rows, source="test")
    assert not tree.complete
    issues = {i["issue"] for i in tree.issues}
    assert ISSUE_MISSING_PARENT in issues
    assert [list(i["path"]) for i in tree.issues if i["issue"] == ISSUE_MISSING_PARENT] \
        == [[2, 0]]

    orphan = tree.frames[tree.index_by_path[(2, 0)]]
    assert orphan.reverted is None, "unknown, not false"
    assert orphan.parent_index is None
    assert tree.index_by_path[(2, 0)] in tree.unresolved_ancestry

    tx = tx_with()
    apply_nesting(tx, tree, source="test")
    assert not tx.call_tree_has_nesting, "an incomplete tree may not claim nesting"
    graph = build_flow_graph(tx)
    assert not graph.native_complete
    assert "no parent pointers" in graph.native_reason or \
           "could not be walked to the root" in graph.native_reason
    assert graph.outcome == "incomplete_graph"
    assert not any(e.asset_kind == "native" for e in graph.edges), \
        "an unknown ancestry must never become a committed transfer"


def test_an_unresolved_ancestry_blocks_the_graph_even_when_nesting_is_claimed():
    """Belt and braces: force the nesting flag on and the unknown still wins."""
    rows = [row([], value=0), row([0], value=100), row([5, 0], value=900)]
    tree = reconstruct(TX, rows, source="test")
    tx = tx_with()
    apply_nesting(tx, tree, source="test")
    tx.call_tree_has_nesting = True           # a caller that lies about completeness
    assert tx.call_tree_ancestry_known
    graph = build_flow_graph(tx)
    assert not graph.native_complete
    assert graph.unresolved_ancestry_frames == 1
    assert "could not be walked to the root" in graph.native_reason


# --- the other validations ----------------------------------------------------

def test_the_root_is_included_exactly_once():
    tree = reconstruct(TX, [row([], value=1), row([0], value=2)], source="test")
    roots = [f for f in tree.frames if f.is_root]
    assert len(roots) == 1
    assert tree.root is not None and tree.root.depth == 0
    assert tree.frame_count == 2


def test_no_root_is_reported_rather_than_promoting_a_child():
    tree = reconstruct(TX, [row([0], value=1), row([1], value=2)], source="test")
    assert tree.status == "unusable"
    assert {i["issue"] for i in tree.issues} == {ISSUE_NO_ROOT}
    assert tree.frames == []


def test_a_duplicate_path_is_reported_and_not_silently_merged():
    tree = reconstruct(TX, [row([], value=0), row([0], value=100),
                            row([0], value=999, to=7)], source="test")
    assert not tree.complete
    assert ISSUE_DUPLICATE_PATH in {i["issue"] for i in tree.issues}
    assert tree.frame_count == 2, "the duplicate did not become a third frame"


def test_a_stated_child_count_that_is_not_met_is_a_missing_child():
    tree = reconstruct(TX, [row([], value=0, subcalls=3), row([0], value=1)], source="test")
    assert not tree.complete
    issue = next(i for i in tree.issues if i["issue"] == ISSUE_MISSING_CHILD)
    assert issue["stated"] == 3 and issue["observed"] == 1


def test_row_order_does_not_decide_nesting():
    """Shuffling the rows must not change a single parent pointer."""
    rows = [row([], value=0), row([0], value=1), row([0, 0], value=2), row([1], value=3)]
    forward = reconstruct(TX, rows, source="test")
    backward = reconstruct(TX, list(reversed(rows)), source="test")
    assert {p: forward.frames[i].parent_index for p, i in forward.index_by_path.items()} == \
           {p: backward.frames[i].parent_index for p, i in backward.index_by_path.items()}
    assert forward.frames[forward.index_by_path[(0, 0)]].depth == 2


def test_depth_comes_from_the_path_length():
    tree = reconstruct(TX, [row([], value=0), row([0], value=0), row([0, 1], value=0),
                            row([0, 1, 2], value=0)], source="test")
    assert sorted(f.depth for f in tree.frames) == [0, 1, 2, 3]
    assert tree.max_depth == 3


# --- the root cross-check -----------------------------------------------------

def test_the_root_is_cross_checked_against_the_local_envelope():
    tree = reconstruct(TX, [row([], value=0, frm=1, to=2)], source="test")
    check = cross_check_root(tree, tx_with())
    assert check["checked"] and check["agrees"], check
    assert "from_address" in check["fields_compared"]


def test_a_root_that_disagrees_is_reported_with_both_values():
    tree = reconstruct(TX, [row([], value=0, frm=1, to=99)], source="test")
    check = cross_check_root(tree, tx_with())
    assert not check["agrees"]
    assert check["mismatches"]["to_address"]["local"] == ADDR(2)
    assert check["mismatches"]["to_address"]["reconstructed"] == ADDR(99)


def test_a_root_error_that_contradicts_the_receipt_is_reported():
    tree = reconstruct(TX, [row([], value=0, error="Reverted")], source="test")
    check = cross_check_root(tree, tx_with(frames_status=1))
    assert "receipt_status" in check["mismatches"]
