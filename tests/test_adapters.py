"""Detector input validation, and the evidence levels kept apart."""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import H, REPO

from txbench.adapters import SPECS, audit, coverage_table, enrichment_plan
from txbench.adapters.base import (BLOCKED, PARTIAL, PROJECT_IMPL, PAPER_REIMPL, SCHEMA_CHECK,
                                   UPSTREAM_DEGRADED, USABLE)

BLOCKSCAN_REPO = REPO / "data" / "upstream" / "detectors" / "blockscan"


def test_all_five_detectors_are_pinned():
    assert set(SPECS) == {"ft_transformer", "blockscan", "blockgpt", "blocklens", "defiguard"}
    for key, spec in SPECS.items():
        assert spec.paper_id, f"{key} has no paper or repository identifier"
        assert spec.requirements, f"{key} declares no required fields"


def test_blockscan_revision_is_pinned_to_a_commit():
    spec = SPECS["blockscan"]
    assert spec.code_revision and len(spec.code_revision) == 40
    assert "preprocess.py" in spec.preprocess_entry


def test_every_record_appears_once_per_detector(records):
    checks = audit(sorted(records.items()))
    assert len(checks) == len(records) * len(SPECS)
    for key, slot in coverage_table(checks).items():
        assert slot["eligible"] == len(records), "a blocked record stays in the denominator"


def test_ft_transformer_is_usable_where_both_views_are(records):
    checks = {c.tx_key: c for c in audit(sorted(records.items()), ["ft_transformer"])}
    assert checks[f"1:{H(1)}"].verdict == USABLE
    assert checks[f"1:{H(1)}"].evidence == PROJECT_IMPL
    # a partial call tree leaves view 1 usable and view 2 not
    partial = checks[f"1:{H(4)}"]
    assert partial.verdict == PARTIAL
    assert partial.notes["view_tx_receipt_usable"] is True
    assert partial.notes["view_full_calls_usable"] is False


def test_missing_record_is_blocked_not_assumed(records):
    checks = {c.tx_key: c for c in audit([(f"1:{H(8)}", None)])}
    for check in checks.values():
        assert check.verdict == BLOCKED
        assert "record" in check.missing_fields


def test_blockscan_is_blocked_on_nesting_with_a_named_reason(records):
    checks = {c.tx_key: c for c in audit(sorted(records.items()), ["blockscan"])}
    check = checks[f"1:{H(1)}"]
    assert check.verdict == BLOCKED
    assert "nested_calls" in check.missing_fields
    assert "flat frame list" in check.missing_reasons["nested_calls"]
    assert check.evidence == SCHEMA_CHECK, "a schema check is never reported as a run"


def test_blockscan_zero_topic_log_is_called_out(records):
    checks = {c.tx_key: c for c in audit(sorted(records.items()), ["blockscan"])}
    check = checks[f"1:{H(4)}"]
    assert "indexes topics[0] unconditionally" in check.missing_reasons["log_typed_topics"]


def test_blockgpt_and_blocklens_are_blocked_on_their_binding_gap(records):
    present = {k: v for k, v in records.items() if v is not None}
    gpt = {c.tx_key: c for c in audit(sorted(present.items()), ["blockgpt"])}
    lens = {c.tx_key: c for c in audit(sorted(present.items()), ["blocklens"])}
    for check in gpt.values():
        assert check.verdict == BLOCKED
        assert "ordered_state_access" in check.missing_fields
    for check in lens.values():
        assert check.verdict == BLOCKED
        assert "opcode_sequence" in check.missing_fields
    # a hash with no record at all is blocked for that reason instead
    missing = {c.tx_key: c for c in audit([(f"1:{H(8)}", None)], ["blockgpt"])}
    assert missing[f"1:{H(8)}"].missing_fields == ["record"]


def test_defiguard_is_a_labelled_reimplementation(records):
    checks = {c.tx_key: c for c in audit(sorted(records.items()), ["defiguard"])}
    good = checks[f"1:{H(6)}"]
    assert good.evidence == PAPER_REIMPL, "our own adapter is never upstream code"
    assert good.verdict == PARTIAL
    assert good.notes["node_feature_count_available"] == 3
    assert good.notes["node_feature_count_required"] == 4
    reverted = checks[f"1:{H(5)}"]
    assert reverted.verdict == BLOCKED
    assert reverted.notes["coverage_outcome"] == "incomplete_graph"
    no_transfer = checks[f"1:{H(9)}"]
    assert no_transfer.notes["coverage_outcome"] == "no_transfers_observed"


def test_enrichment_plan_names_the_source_that_would_unblock_each_field(records):
    plan = enrichment_plan(audit(sorted(records.items())))
    assert "blocklens" in plan and "blockgpt" in plan
    opcode = plan["blocklens"]["missing_profile"]["opcode_sequence"]
    assert opcode["count"] >= 1
    assert "struct log" in (opcode["source"] or "")
    assert plan["blocklens"]["example_tx_keys"], "the hashes needing enrichment are listed"


@pytest.mark.skipif(not BLOCKSCAN_REPO.is_dir(),
                    reason="the pinned BlockScan checkout is not present")
def test_blockscan_entry_point_runs_but_is_labelled_degraded(tx_of):
    from txbench.adapters.blockscan_smoke import run_smoke
    result = run_smoke(tx_of(1), BLOCKSCAN_REPO)
    assert result["ran"] is True, result.get("error")
    assert result["token_count"] > 0
    assert result["fidelity"] == "degraded"
    assert result["nested_calls_supplied"] == 0
    assert "[START]" in result["tokens_head"]
    # the driver needs config files the pinned revision does not ship
    assert result["driver_available"] is False
    assert "config/top_20k_address.txt" in result["driver_missing_files"]


@pytest.mark.skipif(not BLOCKSCAN_REPO.is_dir(), reason="no pinned checkout")
def test_blockscan_conversion_omits_zero_topic_logs_rather_than_crashing(tx_of):
    from txbench.adapters.blockscan_smoke import run_smoke, to_blockscan_record
    record = to_blockscan_record(tx_of(4))
    assert all(item["topics"] for item in record["logs"])
    result = run_smoke(tx_of(4), BLOCKSCAN_REPO)
    assert result["ran"] is True
    assert result["zero_topic_logs_omitted"] == 1
