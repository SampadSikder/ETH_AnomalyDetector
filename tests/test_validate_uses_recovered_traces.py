"""The validator must judge the records a detector would actually receive.

`bench enrich-traces` recovers call trees into the local cache. If the
validator ignores that cache, it reports a detector as blocked on nesting that
has in fact been recovered, and the BlockScan smoke test runs the authors'
preprocessor on a flat record - which proves nothing about whether real
nesting survives the conversion.

These tests go through the real entry point, `cmd_validate_adapters`, against
a temporary root seeded with one cached nested trace. Nothing contacts
BigQuery or an RPC endpoint.
"""
from __future__ import annotations

from argparse import Namespace

import pandas as pd
import pytest

from conftest import FIXTURE_RELEASE, H

from txbench import bq
from txbench.cache import CacheStore
from txbench.enrich import store_rows
from txbench.paths import layout as make_layout

SUBJECT = H(6)                      # four release frames, receipt committed
TABLE = bq.DEFAULT_TABLE


def _args(root, **extra):
    base = dict(root=str(root), config=None, release_dir=str(FIXTURE_RELEASE),
                release_id=None, doi=None, skip_checksums=True, chain_id=None,
                detector=None, blockscan_repo=None, ignore_recovered_traces=False,
                pilot=False)
    base.update(extra)
    return Namespace(**base)


def _row(path, *, frm, to, value=0, error=None, call_type="CALL", subcalls=None, inp=None):
    return {"tx_hash": SUBJECT, "call_path": list(path), "call_type": call_type,
            "from_address": frm, "to_address": to,
            "value": None if value is None else str(value),
            "input": inp, "output": None, "error": error,
            "subcall_count": subcalls, "block_number": 100,
            "block_timestamp": "2024-01-01T00:00:00+00:00"}


#: The subject's four release frames, mirrored so the recovered root agrees
#: with the local transaction and the two descriptions line up frame for frame.
A = "0x1010101010101010101010101010101010101010"
B = "0x2020202020202020202020202020202020202020"
C = "0x3030303030303030303030303030303030303030"
D = "0x3535353535353535353535353535353535353535"
E = "0x3636363636363636363636363636363636363636"
ROOT_INPUT = ("0x38ed1739" + "01" * 64)


@pytest.fixture
def seeded(tmp_path, manifest):
    """A chain of four frames, cached as `enrich-traces` leaves it.

    The release lists these same four frames flat and in this order, so depth
    first pre-order lines the two up; the middle frame reverts, which makes its
    child a success that was undone. Only the root carries input, because the
    warehouse leaves it empty for frames whose bytes live elsewhere - the case
    that must not cost the record the release's own payloads.
    """
    layout = make_layout(tmp_path)
    layout.ensure()
    layout.manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(layout.manifest, index=False)
    rows = [
        _row([], frm=A, to=B, value=0, subcalls=1, inp=ROOT_INPUT),
        _row([0], frm=B, to=C, value=250, subcalls=1),
        _row([0, 0], frm=C, to=D, value=999, subcalls=1, call_type="DELEGATECALL",
             error="Reverted"),                       # the frame that failed
        _row([0, 0, 0], frm=C, to=E, value=None, subcalls=0, call_type="STATICCALL"),
    ]                                                 # succeeded, undone by its parent
    store = CacheStore(layout.cache_dir, layout.cache_index)
    store_rows(store, 1, SUBJECT, TABLE, rows)
    return tmp_path, layout


def _coverage(layout, detector):
    table = pd.read_parquet(layout.adapters_dir / "cohort" / "compatibility_table.parquet")
    rows = table[(table.detector == detector) & (table.tx_key == f"1:{SUBJECT}")]
    assert len(rows) == 1, f"expected one {detector} row for the subject"
    row = rows.iloc[0]
    return {
        "verdict": row.verdict,
        "available": set(str(row.required_fields_available or "").split(";")) - {""},
        "missing": set(str(row.required_fields_missing or "").split(";")) - {""},
    }


def test_the_validator_uses_a_cached_nested_trace(seeded):
    """The regression: nesting recovered into the cache must reach the audit."""
    from txbench.cli import cmd_validate_adapters
    root, layout = seeded
    assert cmd_validate_adapters(_args(root)) == 0

    blockscan = _coverage(layout, "blockscan")
    assert "nested_calls" in blockscan["available"], \
        "the cached tree supplies the recursive calls array"
    assert "nested_calls" not in blockscan["missing"]

    defiguard = _coverage(layout, "defiguard")
    assert "reverted_ancestry" in defiguard["available"], \
        "a real parent pointer is what resolves ancestry"

    blockgpt = _coverage(layout, "blockgpt")
    assert "nested_structure" in blockgpt["available"]


def test_without_the_cache_the_same_records_are_blocked_on_nesting(seeded):
    """The negative control, so the test above cannot pass for another reason."""
    from txbench.cli import cmd_validate_adapters
    root, layout = seeded
    assert cmd_validate_adapters(_args(root, ignore_recovered_traces=True)) == 0

    blockscan = _coverage(layout, "blockscan")
    assert "nested_calls" in blockscan["missing"]
    defiguard = _coverage(layout, "defiguard")
    assert "reverted_ancestry" in defiguard["missing"]


def test_independent_requirements_are_not_satisfied_by_a_recovered_tree(seeded):
    """A call tree is not opcodes, decoded arguments or an account kind."""
    from txbench.cli import cmd_validate_adapters
    root, layout = seeded
    cmd_validate_adapters(_args(root))

    assert "node_type" in _coverage(layout, "defiguard")["missing"]
    assert "frame_decoded_args" in _coverage(layout, "blockscan")["missing"]
    assert _coverage(layout, "blocklens")["missing"] >= {"opcode_sequence", "step_operands"}
    assert "ordered_state_access" in _coverage(layout, "blockgpt")["missing"]
    assert _coverage(layout, "blockgpt")["verdict"] == "blocked", \
        "nesting alone must not promote a detector that needs state access"


def test_release_payload_bytes_are_not_lost_when_a_tree_is_applied(seeded):
    """Applying a tree replaces the frame list, which must not drop real data.

    The cached rows carry no input for the children - as the warehouse leaves a
    CREATE or a SELFDESTRUCT - so whatever the release supplied has to be
    carried back onto the recovered frames.
    """
    from txbench.cli import _assembled, _context
    root, layout = seeded
    args = _args(root)
    cfg, resolved = _context(args)

    before = {k: tx for k, tx in _assembled(cfg, resolved, _args(root,
              ignore_recovered_traces=True))[2] if tx is not None}
    after = {k: tx for k, tx in _assembled(cfg, resolved, args)[2] if tx is not None}

    key = f"1:{SUBJECT}"
    assert after[key].call_tree_has_nesting, "the subject was enriched"
    had = sum(1 for f in before[key].frames if f.input_hex is not None)
    kept = sum(1 for f in after[key].frames if f.input_hex is not None)
    assert kept >= had, (f"{had - kept} frame(s) lost the input bytes the release "
                         "supplied when the tree was applied")


def test_a_recovered_tree_whose_root_disagrees_is_refused(tmp_path, manifest):
    """A record describing a different transaction is worse than a flat one."""
    from txbench.cli import cmd_validate_adapters
    layout = make_layout(tmp_path)
    layout.ensure()
    manifest.to_parquet(layout.manifest, index=False)
    store = CacheStore(layout.cache_dir, layout.cache_index)
    wrong = "0x9999999999999999999999999999999999999999"
    store_rows(store, 1, SUBJECT, TABLE, [
        _row([], frm=wrong, to=wrong, value=0, subcalls=1),
        _row([0], frm=wrong, to=wrong, value=0),
    ])
    assert cmd_validate_adapters(_args(tmp_path)) == 0
    assert "nested_calls" in _coverage(layout, "blockscan")["missing"], \
        "a tree that cross-checks against a different transaction is not applied"
