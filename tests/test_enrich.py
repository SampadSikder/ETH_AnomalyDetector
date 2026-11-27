"""The backfill's offline halves: query construction, budget, cache reuse.

Nothing here contacts BigQuery. The query is checked as text, the budget as
arithmetic, and the cache by storing rows and asking a second run what it
would fetch - which is the acceptance check that a completed pilot costs
nothing to repeat.
"""
from __future__ import annotations

import pytest

from txbench import bq
from txbench.cache import CacheStore
from txbench.enrich import (CACHE_METHOD, Pilot, cached_rows, partition_cached, rows_to_traces,
                            store_rows, tracer_config_for)

from conftest import ADDR, H

TABLE = bq.DEFAULT_TABLE


def description(*, columns=None, partition="time", partition_field="block_timestamp",
                partition_type="TIMESTAMP", table_type="TABLE", granularity="DAY"):
    cols = columns if columns is not None else {
        "transaction_hash": "STRING", "trace_address": "STRING", "call_type": "STRING",
        "from_address": "STRING", "to_address": "STRING", "value": "BIGNUMERIC",
        "input": "STRING", "output": "STRING", "error": "STRING", "status": "INTEGER",
        "subtraces": "INTEGER", "block_number": "INTEGER", "block_timestamp": "TIMESTAMP",
    }
    out = bq.TableDescription(table=TABLE, columns=dict(cols))
    out.modes = {name: "NULLABLE" for name in cols}
    out.table_type = table_type
    out.partition_kind = partition
    out.partition_field = partition_field if partition == "time" else None
    out.partition_type = granularity if partition == "time" else None
    if partition_field in out.columns:
        out.columns[partition_field] = partition_type
    return out


# --- what gets selected -------------------------------------------------------

def _coverage_fixture(tmp_path):
    import pandas as pd
    adapters = tmp_path / "adapters" / "cohort"
    adapters.mkdir(parents=True)
    pd.DataFrame({
        "detector": ["defiguard"] * 3 + ["blockgpt"],
        "tx_key": ["k1", "k2", "k3", "k1"],
        "verdict": ["blocked", "partial", "usable", "blocked"],
        "missing_field_reason": ['{"native_flows": "3 frame(s) failed"}', "{}", "{}", "{}"],
    }).to_parquet(adapters / "compatibility_table.parquet", index=False)
    manifest = pd.DataFrame({
        "tx_key": ["k1", "k2", "k3"], "tx_hash": [H(1), H(2), H(3)],
        "label": [1, 0, 0], "attack_class": ["exploit", None, None],
        "block_number": [100, 200, 300],
        "block_timestamp": [1500000000, 1600000000, 1700000000],
        "split_temporal": ["test", "train", "train"],
        "group_incident": ["i1", "i2", "i3"],
    })
    from types import SimpleNamespace
    return SimpleNamespace(adapters_dir=tmp_path / "adapters"), manifest


def test_the_default_selection_is_still_only_the_blocked_transactions(tmp_path):
    """Extending the command must not quietly widen the pilot."""
    from txbench.enrich import select_pilot
    layout, manifest = _coverage_fixture(tmp_path)
    assert select_pilot(layout, manifest, detector="defiguard").hashes == [H(1)]


def test_selecting_all_takes_every_transaction_whatever_its_verdict(tmp_path):
    from txbench.enrich import select_pilot
    layout, manifest = _coverage_fixture(tmp_path)
    everything = select_pilot(layout, manifest, detector="defiguard", select="all")
    assert sorted(everything.hashes) == sorted([H(1), H(2), H(3)])
    assert everything.summary()["unique_hashes"] == 3
    assert len(everything.blocks_by_date()) == 3, "one UTC day per block here"


def test_an_unknown_selection_is_refused(tmp_path):
    from txbench.enrich import select_pilot
    layout, manifest = _coverage_fixture(tmp_path)
    with pytest.raises(Exception) as caught:
        select_pilot(layout, manifest, detector="defiguard", select="everything")
    assert "expected 'blocked' or 'all'" in str(caught.value)


# --- schema resolution --------------------------------------------------------

def test_every_needed_role_is_matched_against_the_real_schema():
    resolved = bq.resolve_columns(description())
    assert resolved["usable_for_call_tree"]
    assert resolved["mapping"]["call_path"] == "trace_address"
    assert resolved["mapping"]["tx_hash"] == "transaction_hash"
    assert resolved["missing_flow_roles"] == []


def test_a_table_without_a_call_path_is_refused_before_any_scan():
    """The one check that must happen before bytes move."""
    flat = description(columns={
        "transaction_hash": "STRING", "from_address": "STRING", "to_address": "STRING",
        "value": "BIGNUMERIC", "block_timestamp": "TIMESTAMP"})
    resolved = bq.resolve_columns(flat)
    assert not resolved["usable_for_call_tree"]
    assert "call_path" in resolved["missing_required_roles"]
    with pytest.raises(bq.NoCallPath) as caught:
        bq.build_query(flat, resolved, hashes=[H(1)], dates=["2025-01-01"])
    assert "call-path" in str(caught.value)


def test_a_view_without_partition_metadata_is_bounded_explicitly_not_refused():
    """The real Ethereum traces source is a view, which declares no partitioning.

    Treating that absence as "unpartitioned" would refuse the only source
    there is. The bound is written out explicitly against a real timestamp
    column instead, and the dry run is what reports the cost.
    """
    view = description(partition="none", table_type="VIEW")
    assert view.as_dict()["partition_metadata_available"] is False
    bound = bq.resolve_time_bound(view)
    assert bound["column"] == "block_timestamp"
    assert bound["basis"] == "view_without_partition_metadata"
    assert bound["granularity"] is None, "unknown granularity, not assumed"

    query = bq.build_query(view, bq.resolve_columns(view), hashes=[H(1)],
                           dates=["2025-01-01"])
    assert 'block_timestamp >= TIMESTAMP("2025-01-01")' in query.sql
    assert 'block_timestamp < TIMESTAMP("2025-01-02")' in query.sql
    assert query.time_bound["basis"] == "view_without_partition_metadata"


def test_a_source_with_no_timestamp_column_at_all_is_refused():
    """What must still be refused: a scan with no time bound to put on it."""
    timeless = description(columns={
        "transaction_hash": "STRING", "trace_address": "STRING", "value": "BIGNUMERIC"})
    with pytest.raises(Exception) as caught:
        bq.build_query(timeless, bq.resolve_columns(timeless), hashes=[H(1)],
                       dates=["2025-01-01"])
    assert "unbounded scan" in str(caught.value)


def test_nested_record_fields_fill_the_roles_they_hold():
    """The public view keeps caller, callee, value and input inside `action`."""
    view = description(partition="none", table_type="VIEW", columns={
        "transaction_hash": "STRING", "trace_address": "INTEGER", "trace_type": "STRING",
        "subtrace_count": "INTEGER", "error": "STRING", "block_number": "INTEGER",
        "block_timestamp": "TIMESTAMP", "action": "RECORD",
        "action.from_address": "STRING", "action.to_address": "STRING",
        "action.call_type": "STRING", "action.value_lossless": "STRING",
        "action.value": "BIGNUMERIC", "action.input": "STRING",
        "result": "RECORD", "result.output": "STRING"})
    resolved = bq.resolve_columns(view)
    assert resolved["usable_for_call_tree"]
    assert resolved["missing_flow_roles"] == []
    assert resolved["mapping"]["from_address"] == "action.from_address"
    assert resolved["mapping"]["output"] == "result.output"
    assert resolved["mapping"]["value"] == "action.value_lossless", \
        "the lossless string form, because wei does not survive a float"
    assert resolved["mapping"]["subcall_count"] == "subtrace_count"

    query = bq.build_query(view, resolved, hashes=[H(1)], dates=["2025-01-01"])
    assert "action.from_address AS from_address" in query.sql
    assert "result.output AS output" in query.sql
    assert "action.gas" not in query.sql, "nothing the reconstruction does not need"


def test_exact_block_numbers_narrow_each_span():
    """A transaction sits in one block, and the manifest records which."""
    desc = description()
    query = bq.build_query(desc, bq.resolve_columns(desc), hashes=[H(1), H(2)],
                           dates=["2025-01-01", "2025-06-02"],
                           blocks_by_date={"2025-01-01": [100, 101], "2025-06-02": [900]})
    assert "block_number IN (100, 101)" in query.sql
    assert "block_number IN (900)" in query.sql
    assert query.blocks_narrowed == 3
    assert query.sql.count("block_number IN") == 2, "one narrowing per span"


def test_block_numbers_for_a_day_that_was_not_requested_are_an_error():
    desc = description()
    with pytest.raises(Exception) as caught:
        bq.build_query(desc, bq.resolve_columns(desc), hashes=[H(1)],
                       dates=["2025-01-01"], blocks_by_date={"2025-01-02": [5]})
    assert "not in the request" in str(caught.value)


def test_a_monthly_partition_widens_each_day_to_its_whole_month():
    """What a monthly partition costs, the predicate may as well say.

    Two days in one month become one span, not two.
    """
    monthly = description(granularity="MONTH")
    query = bq.build_query(monthly, bq.resolve_columns(monthly), hashes=[H(1)],
                           dates=["2025-03-07", "2025-03-30", "2025-12-31"],
                           blocks_by_date={"2025-03-07": [1], "2025-03-30": [2],
                                           "2025-12-31": [3]})
    assert len(query.spans) == 2, "March once, December once"
    assert 'block_timestamp >= TIMESTAMP("2025-03-01")' in query.sql
    assert 'block_timestamp < TIMESTAMP("2025-04-01")' in query.sql
    assert 'block_timestamp < TIMESTAMP("2026-01-01")' in query.sql, "a year boundary"
    assert "block_number IN (1, 2)" in query.sql, "both March blocks in one clause"


# --- the query ----------------------------------------------------------------

def test_the_query_prunes_one_clause_per_day_and_parameterizes_the_hashes():
    desc = description()
    resolved = bq.resolve_columns(desc)
    dates = ["2025-03-07", "2017-07-19", "2025-03-07"]
    query = bq.build_query(desc, resolved, hashes=[H(1), H(2)], dates=dates)

    assert query.dates == ["2017-07-19", "2025-03-07"], "a shared day is scanned once"
    assert query.sql.count("block_timestamp >=") == 2
    assert 'TIMESTAMP("2017-07-19")' in query.sql
    assert "IN UNNEST(@hashes)" in query.sql, "the hash list stays a parameter"
    assert H(1) not in query.sql, "no hash is interpolated into the SQL"
    # Only the needed columns, never a star.
    assert "SELECT *" not in query.sql
    for column in ("transaction_hash", "trace_address", "call_type", "value"):
        assert column in query.sql
    assert "gas_price" not in query.sql


def test_the_query_selects_only_mapped_columns_and_aliases_them_to_roles():
    desc = description()
    query = bq.build_query(desc, bq.resolve_columns(desc), hashes=[H(1)],
                           dates=["2026-02-20"])
    assert "transaction_hash AS tx_hash" in query.sql
    assert "trace_address AS call_path" in query.sql
    assert "subtraces AS subcall_count" in query.sql


@pytest.mark.parametrize("bad", ["2025-3-7", "07/19/2017", "2025-02-30", "yesterday",
                                 "2025-01-01 OR 1=1"])
def test_only_a_real_iso_date_reaches_the_partition_predicate(bad):
    """These become SQL literals, so nothing else may pass."""
    desc = description()
    with pytest.raises(Exception):
        bq.build_query(desc, bq.resolve_columns(desc), hashes=[H(1)], dates=[bad])


def test_a_date_partition_column_is_bounded_as_a_half_open_range():
    desc = description(partition_field="block_date", partition_type="DATE")
    desc.columns["block_date"] = "DATE"
    resolved = bq.resolve_columns(desc)
    query = bq.build_query(desc, resolved, hashes=[H(1)], dates=["2025-05-05"])
    assert 'block_date >= DATE "2025-05-05"' in query.sql
    assert 'block_date < DATE "2025-05-06"' in query.sql


def test_an_estimate_bigquery_will_not_give_is_unknown_not_zero():
    """Folding a missing estimate to 0 would read as "this query is free"."""
    class _Job:
        total_bytes_processed = None
        cache_hit = False

    class _Client:
        def query(self, sql, job_config=None):
            return _Job()

    desc = description()
    query = bq.build_query(desc, bq.resolve_columns(desc), hashes=[H(1)],
                           dates=["2025-01-01"])
    priced = bq.dry_run(_Client(), query)
    assert priced["estimated_bytes_processed"] is None
    assert priced["estimate_available"] is False
    assert "unknown" in priced["estimate_note"]


def test_an_unknown_estimate_cannot_run_without_an_explicit_budget(tmp_path):
    unbudgeted = bq.Budget.load(tmp_path / "b.json", None)
    with pytest.raises(bq.BudgetExhausted) as caught:
        unbudgeted.ceiling_for(None)
    assert "--budget-bytes" in str(caught.value)
    budgeted = bq.Budget.load(tmp_path / "c.json", 5_000)
    assert budgeted.ceiling_for(None) == 5_000, "the budget is the only ceiling left"


def test_warehouse_objects_survive_the_cache_as_the_same_values():
    """BigQuery returns datetime, Decimal and bytes; the cache writes JSON.

    This is the crash that cost a paid-for query: 8.4 GB were billed, the rows
    came back, and caching them raised on the first `block_timestamp`. The
    round trip must also be lossless, or a re-run would not be a re-run.
    """
    from datetime import datetime, timezone
    from decimal import Decimal
    import json as _json
    from txbench.enrich import jsonable_rows

    raw = [{
        "tx_hash": H(1),
        "call_path": [1, 0],
        "block_timestamp": datetime(2025, 3, 7, 12, 30, 5, tzinfo=timezone.utc),
        "value": Decimal("1000000000000000000"),
        "input": b"\x01\xab",
        "subcall_count": 2,
        "error": None,
    }]
    rows = jsonable_rows(raw)
    # It must actually serialise - that is the thing that failed.
    assert _json.loads(_json.dumps(rows, sort_keys=True)) == rows
    assert rows[0]["block_timestamp"] == "2025-03-07T12:30:05+00:00"
    assert rows[0]["value"] == "1000000000000000000", "a digit string, never a float"
    assert rows[0]["input"] == "0x01ab"
    assert rows[0]["call_path"] == [1, 0], "the repeated path stays a list of ints"

    # And the reconstruction reads the same values either way.
    fresh = rows_to_traces(rows)[H(1)][0]
    assert fresh.value_wei == 10**18, "wei survives, exactly"
    assert fresh.path == (1, 0)
    assert fresh.input_hex == "0x01ab"


def test_a_lossless_wei_string_is_not_rounded_through_a_float():
    """10**18 + 1 differs from 10**18 only beyond a float's mantissa."""
    from decimal import Decimal
    from txbench.enrich import jsonable
    assert jsonable(Decimal("1000000000000000001")) == "1000000000000000001"
    assert rows_to_traces([{"tx_hash": H(1), "call_path": "0",
                            "value": jsonable(Decimal("1000000000000000001"))}
                           ])[H(1)][0].value_wei == 10**18 + 1


def test_storing_rows_normalises_whatever_it_is_handed(tmp_path):
    """The cache is the thing that cannot take a datetime, so it converts."""
    from datetime import datetime, timezone
    from txbench.enrich import store_rows
    store = CacheStore(tmp_path / "blobs", tmp_path / "index.sqlite")
    store_rows(store, 1, H(7), TABLE,
               [{"tx_hash": H(7), "call_path": [],
                 "block_timestamp": datetime(2024, 1, 1, tzinfo=timezone.utc)}])
    back = cached_rows(store, 1, H(7), TABLE)
    assert back == [{"tx_hash": H(7), "call_path": [],
                     "block_timestamp": "2024-01-01T00:00:00+00:00"}]


# --- the targeted init backfill -----------------------------------------------

def test_the_init_query_asks_for_init_and_not_for_runtime_code():
    """`result.code` is excluded because it cannot reach the token stream.

    Varying it through the BlockScan adapter produced a byte-identical result,
    so fetching it would bill bytes for something no reader can see.
    """
    desc = description(partition="none", table_type="VIEW", columns={
        "transaction_hash": "STRING", "trace_address": "INTEGER", "trace_type": "STRING",
        "block_number": "INTEGER", "block_timestamp": "TIMESTAMP",
        "action": "RECORD", "action.init": "STRING", "action.input": "STRING",
        "action.from_address": "STRING", "action.to_address": "STRING",
        "action.value_lossless": "STRING", "result": "RECORD",
        "result.output": "STRING", "result.code": "STRING"})
    query = bq.build_init_query(desc, bq.resolve_columns(desc), hashes=[H(1)],
                                dates=["2025-01-01"], blocks_by_date={"2025-01-01": [7]})
    assert "action.init AS init" in query.sql
    assert "result.code" not in query.sql, "runtime code changes no token; do not pay for it"
    assert "trace_type = 'create'" in query.sql, "only creation frames"
    assert "action.input" not in query.sql
    assert "block_number IN (7)" in query.sql
    assert "IN UNNEST(@hashes)" in query.sql


def test_a_table_without_action_init_cannot_supply_init_bytes():
    desc = description()
    with pytest.raises(Exception) as caught:
        bq.build_init_query(desc, bq.resolve_columns(desc), hashes=[H(1)],
                            dates=["2025-01-01"])
    assert "action.init" in str(caught.value)


def test_only_creation_frames_below_the_root_are_selected_as_gaps(tmp_path):
    """A root creation frame needs no query: the transaction input is its init."""
    from txbench.enrich import select_init_gaps
    store = CacheStore(tmp_path / "blobs", tmp_path / "index.sqlite")
    store_rows(store, 1, H(3), TABLE, [
        {"tx_hash": H(3), "call_path": [], "trace_type": "create", "input": None},
        {"tx_hash": H(3), "call_path": [0], "trace_type": "create", "input": None},
        {"tx_hash": H(3), "call_path": [1], "trace_type": "create", "input": "0x6080"},
        {"tx_hash": H(3), "call_path": [2], "trace_type": "call", "input": None},
        {"tx_hash": H(3), "call_path": [3], "trace_type": "suicide", "input": None},
    ])
    gaps = select_init_gaps(store, 1, TABLE, [H(3)])
    assert gaps == {H(3).lower(): [[0]]}, \
        "the root, a filled frame, a call and a selfdestruct are all left alone"


def test_patching_init_fills_only_what_is_empty_and_never_appends(tmp_path):
    from txbench.enrich import patch_cached_init
    store = CacheStore(tmp_path / "blobs", tmp_path / "index.sqlite")
    store_rows(store, 1, H(4), TABLE, [
        {"tx_hash": H(4), "call_path": [0], "trace_type": "create", "input": None},
        {"tx_hash": H(4), "call_path": [1], "trace_type": "create", "input": "0xkeep"},
    ])
    out = patch_cached_init(store, 1, TABLE, [
        {"tx_hash": H(4), "call_path": [0], "init": "0x60806040"},
        {"tx_hash": H(4), "call_path": [1], "init": "0xoverwrite"},
        {"tx_hash": H(4), "call_path": [9], "init": "0xstranger"},
    ])
    assert out["frames_patched"] == 1
    assert out["frames_unmatched"] == 1, "a path the reconstruction never saw is reported"
    rows = {tuple(r["call_path"]): r for r in cached_rows(store, 1, H(4), TABLE)}
    assert len(rows) == 2, "no row was appended"
    assert rows[(0,)]["input"] == "0x60806040"
    assert rows[(1,)]["input"] == "0xkeep", "an existing value is left alone"


def test_a_frame_already_filled_by_the_release_is_not_fetched_again():
    """The cache stays null where the release supplied the bytes, so the gap
    has to be measured on the assembled record, not on the cached rows."""
    from txbench.enrich import hashes_missing_creation_input
    from txbench.trace_model import CallFrame, RECEIPT_OK, NormalizedTx

    def tx_with(frames):
        return NormalizedTx(chain_id=1, tx_hash=H(1), from_address=ADDR(1),
                            to_address=ADDR(2), value_wei=0, receipt_status=1,
                            receipt_status_source=RECEIPT_OK, logs_available=True,
                            logs=[], frames=frames)

    filled = tx_with([CallFrame(index=0, call_type="CREATE", input_hex="0x6080")])
    empty = tx_with([CallFrame(index=0, call_type="CREATE", input_hex=None)])
    a_call = tx_with([CallFrame(index=0, call_type="CALL", input_hex=None)])
    assert hashes_missing_creation_input([("k", filled)]) == []
    assert hashes_missing_creation_input([("k", empty)]) == [H(1).lower()]
    assert hashes_missing_creation_input([("k", a_call)]) == [], \
        "a plain call with no calldata is not a creation gap"


# --- the budget ---------------------------------------------------------------

def test_an_unconfigured_budget_still_caps_a_job(tmp_path):
    budget = bq.Budget.load(tmp_path / "b.json", None)
    assert not budget.configured
    assert budget.remaining is None
    ceiling = budget.ceiling_for(100 * 1024 * 1024)
    assert ceiling >= 100 * 1024 * 1024, "the ceiling never sits below the estimate"


def test_the_ceiling_is_the_remaining_budget_not_the_estimate(tmp_path):
    budget = bq.Budget.load(tmp_path / "b.json", 1_000_000)
    assert budget.ceiling_for(10_000) == 1_000_000, \
        "a dry run can under-report, so the ceiling is what is left"


def test_a_job_larger_than_the_remaining_budget_is_refused(tmp_path):
    budget = bq.Budget.load(tmp_path / "b.json", 1_000)
    with pytest.raises(bq.BudgetExhausted):
        budget.ceiling_for(5_000)


def test_spending_persists_across_runs(tmp_path):
    path = tmp_path / "b.json"
    first = bq.Budget.load(path, 1_000_000)
    first.charge(billed=400_000, estimated=390_000, label="pilot", job_id="job-1")
    first.save()
    second = bq.Budget.load(path, None)
    assert second.spent_bytes == 400_000
    assert second.remaining == 600_000
    assert second.total_bytes == 1_000_000, "the ledger remembers its own ceiling"
    with pytest.raises(bq.BudgetExhausted):
        second.ceiling_for(700_000)


def test_an_exhausted_budget_names_what_was_spent(tmp_path):
    budget = bq.Budget.load(tmp_path / "b.json", 100)
    budget.charge(billed=100, estimated=100, label="pilot", job_id=None)
    with pytest.raises(bq.BudgetExhausted) as caught:
        budget.ceiling_for(1)
    assert "budget is spent" in str(caught.value)


# --- rows -> traces -----------------------------------------------------------

def test_rows_are_read_by_role_and_grouped_by_transaction():
    rows = [
        {"tx_hash": H(1).upper(), "call_path": "", "call_type": "CALL",
         "from_address": ADDR(1), "to_address": ADDR(2), "value": "100", "subcall_count": 1},
        {"tx_hash": H(1), "call_path": "0", "call_type": "STATICCALL",
         "from_address": ADDR(2), "to_address": ADDR(3), "value": None},
        {"tx_hash": H(2), "call_path": "0,1", "call_type": "CALL", "value": "0x10"},
    ]
    grouped = rows_to_traces(rows)
    assert set(grouped) == {H(1), H(2)}, "hashes are lower-cased"
    assert len(grouped[H(1)]) == 2
    root = next(r for r in grouped[H(1)] if r.path == ())
    assert root.value_wei == 100 and root.subcall_count == 1
    assert grouped[H(2)][0].value_wei == 16, "a hex value is decoded"
    assert grouped[H(2)][0].path == (0, 1)


@pytest.mark.parametrize("row,expect_error", [
    ({"error": "Reverted"}, True),
    ({"error": "out of gas"}, True),
    ({"error": None, "status": 1}, False),
    ({"error": "", "status": 0}, True),
    ({"status": "false"}, True),
    ({"status": None, "error": None}, False),
])
def test_a_frame_failure_is_read_from_either_convention(row, expect_error):
    full = {"tx_hash": H(3), "call_path": "", **row}
    trace = rows_to_traces([full])[H(3)][0]
    assert trace.failed is expect_error


# --- cache reuse --------------------------------------------------------------

def test_a_completed_pilot_refetches_nothing(tmp_path):
    """The acceptance check: a second run has no misses, so it queries nothing."""
    import pandas as pd
    store = CacheStore(tmp_path / "cache", tmp_path / "index.sqlite")
    frame = pd.DataFrame({
        "tx_hash": [H(1), H(2)], "tx_key": [f"1:{H(1)}", f"1:{H(2)}"],
        "label": [1, 0], "split_temporal": ["train", "test"],
        "block_number": [100, 200], "block_date": ["2025-01-01", "2025-01-02"],
        "blocked_reason_kind": ["failed_frames", "failed_frames"]})
    pilot = Pilot(frame=frame, source=tmp_path / "coverage.parquet", expected=2)

    hits, misses = partition_cached(pilot, store, 1, TABLE)
    assert hits == [] and misses == [H(1), H(2)], "a first run must fetch both"

    for tx_hash in misses:
        store_rows(store, 1, tx_hash, TABLE,
                   [{"tx_hash": tx_hash, "call_path": "", "call_type": "CALL"}])

    hits, misses = partition_cached(pilot, store, 1, TABLE)
    assert hits == [H(1), H(2)] and misses == [], "a second run fetches nothing"
    assert cached_rows(store, 1, H(1), TABLE)[0]["call_type"] == "CALL"
    store.close()


def test_an_empty_result_is_cached_so_a_no_match_is_not_refetched(tmp_path):
    store = CacheStore(tmp_path / "cache", tmp_path / "index.sqlite")
    store_rows(store, 1, H(9), TABLE, [])
    rows = cached_rows(store, 1, H(9), TABLE)
    assert rows == [], "an answered 'no rows' is a hit, not a miss"
    store.close()


def test_the_cache_key_is_neutral_and_names_the_source(tmp_path):
    from txbench.enrich import cache_ref
    ref = cache_ref(1, H(4), TABLE)
    path = str(ref.relative_path())
    assert CACHE_METHOD in path
    assert tracer_config_for(TABLE) in path
    assert "attack" not in path and "benign" not in path, "no class information in the path"


# --- credentials --------------------------------------------------------------

def test_a_missing_client_or_credential_fails_with_an_actionable_message():
    """This environment has no ADC, so the message is what a user will see."""
    with pytest.raises(bq.BigQueryUnavailable) as caught:
        bq.load_client(project=None)
    message = str(caught.value)
    assert ("application-default login" in message or "not installed" in message
            or "no Google Cloud project" in message)


# --- end to end on a real blocked shape ---------------------------------------

def _warehouse_rows(tx_hash, layout):
    """Warehouse-shaped rows for the fixture's reverted-subcall transaction.

    The paths are the only thing the warehouse would add: `(1,)` is the call
    that moved 500 wei and `(1, 0)` makes it a *child* of the reverted frame,
    which is the case that must not produce an edge.
    """
    return [{"tx_hash": tx_hash, "call_path": path, "call_type": kind,
             "from_address": frm, "to_address": to, "value": value, "error": error,
             "status": 1 if not error else 0, "subcall_count": None}
            for path, kind, frm, to, value, error in layout]


def test_a_blocked_transaction_is_unblocked_when_paths_arrive(tx_of):
    """The whole point of the backfill, on the fixture's blocked shape.

    The root's own fields must match the local envelope, or the cross-check
    rejects the rows - so the root row is built from the record itself.
    """
    from txbench.enrich import rebuild_one
    from txbench.flows import build_flow_graph

    tx = tx_of(5)
    before = build_flow_graph(tx)
    assert not before.native_complete, "the fixture starts blocked"
    assert "no parent pointers" in before.native_reason

    # The reverted frame is a sibling, so the 500-wei call did commit.
    rows = _warehouse_rows(tx.tx_hash, [
        ("", "CALL", tx.from_address, tx.to_address, str(tx.value_wei or 0), None),
        ("0", "CALL", ADDR(2), ADDR(3), "500", None),
        ("1", "CALL", ADDR(2), ADDR(4), "0", "execution reverted"),
    ])
    out = rebuild_one(tx_of(5), rows, tx_key=f"1:{H(5)}", tx_hash=H(5), table=TABLE)
    assert out.tree_complete, out.tree.issues
    assert out.graph_rebuilt and out.ancestry_resolved
    assert out.root_check["agrees"], out.root_check
    assert out.missing_reason is None, out.missing_reason
    assert out.graph_after["native_edges"] == 1, "the committed sibling moved value"
    assert out.graph_after["excluded_reverted_frames"] == 0
    assert out.frame_count == 3


def test_a_committed_looking_call_under_a_reverted_parent_is_dropped(tx_of):
    """Same rows, one path changed: now the 500 wei never moved."""
    from txbench.enrich import rebuild_one

    tx = tx_of(5)
    rows = _warehouse_rows(tx.tx_hash, [
        ("", "CALL", tx.from_address, tx.to_address, str(tx.value_wei or 0), None),
        ("0", "CALL", ADDR(2), ADDR(4), "0", "execution reverted"),
        ("0,0", "CALL", ADDR(4), ADDR(3), "500", None),
    ])
    out = rebuild_one(tx_of(5), rows, tx_key=f"1:{H(5)}", tx_hash=H(5), table=TABLE)
    assert out.tree_complete and out.graph_rebuilt
    assert out.graph_after["native_edges"] == 0, \
        "the only value-moving frame was undone by its parent"
    assert out.graph_after["excluded_reverted_frames"] == 1


def test_a_hash_with_no_warehouse_rows_still_gets_a_coverage_row(tx_of, tmp_path):
    from txbench.enrich import rebuild_one, write_enrichment_report
    import pandas as pd

    frame = pd.DataFrame({
        "tx_hash": [H(5)], "tx_key": [f"1:{H(5)}"], "label": [1],
        "split_temporal": ["train"], "block_number": [10], "block_date": ["2025-01-01"],
        "blocked_reason_kind": ["failed_frames"]})
    pilot = Pilot(frame=frame, source=tmp_path / "c.parquet", expected=1)
    empty = rebuild_one(tx_of(5), [], tx_key=f"1:{H(5)}", tx_hash=H(5), table=TABLE)
    assert "no BigQuery match" in empty.missing_reason

    written = write_enrichment_report(pilot, [empty], tmp_path, extras={})
    rows = pd.read_parquet(written["coverage_table"])
    assert len(rows) == 1, "one row per requested hash, matched or not"
    assert rows.iloc[0].defiguard_graph_rebuilt is False or \
        not bool(rows.iloc[0].defiguard_graph_rebuilt)
    assert rows.iloc[0].frame_count == 0
