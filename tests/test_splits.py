"""Splits: promises checked, leakage caught, validation carved out of train."""
from __future__ import annotations

import pandas as pd

from txbench.splits import (TEST, TRAIN, VALIDATION, audit_split, corrected_split,
                            make_validation_partition, verify)


def test_every_fixture_split_is_audited(manifest):
    report = verify(manifest, primary="temporal", secondary=["protocol"],
                    metadata_only=["cross_chain"], validation_fraction=0.2)
    assert set(report["upstream_split_columns"]) >= {"split_temporal", "split_incident"}
    assert report["primary"] == "split_temporal"


def test_validation_comes_out_of_training_only(manifest):
    partitions, meta = make_validation_partition(manifest, "split_temporal", fraction=0.4)
    assignment = {row.tx_key: row.partition for row in partitions.itertuples(index=False)}
    upstream = {row.tx_key: row.split_temporal for row in manifest.itertuples(index=False)}
    for key, partition in assignment.items():
        if partition == VALIDATION:
            assert upstream[key] == TRAIN, "validation may only take training rows"
        if upstream[key] == TEST:
            assert partition == TEST, "a test row is never reassigned"
    assert meta["validation_rows"] >= 1


def test_validation_keeps_whole_incident_groups(manifest):
    partitions, _ = make_validation_partition(manifest, "split_temporal", fraction=0.5)
    assignment = {row.tx_key: row.partition for row in partitions.itertuples(index=False)}
    groups = {}
    for row in manifest.itertuples(index=False):
        groups.setdefault(row.group_incident, set()).add(assignment.get(row.tx_key))
    for group, parts in groups.items():
        parts.discard(None)
        assert not ({TRAIN, VALIDATION} <= parts), \
            f"incident {group} straddles train and validation"


def test_leakage_is_detected_and_a_corrected_split_is_published():
    # One incident with transactions on both sides of the boundary: a leak.
    rows = [
        {"tx_key": "1:0xa", "tx_hash": "0xa", "label": 1, "block_timestamp": "2024-01-01T00:00:00Z",
         "split_temporal": TRAIN, "group_incident": "inc-1", "group_protocol": "p",
         "group_address": "0x1"},
        {"tx_key": "1:0xb", "tx_hash": "0xb", "label": 1, "block_timestamp": "2025-02-01T00:00:00Z",
         "split_temporal": TEST, "group_incident": "inc-1", "group_protocol": "p",
         "group_address": "0x1"},
    ]
    frame = pd.DataFrame(rows)
    audit = audit_split(frame, "split_temporal")
    assert audit.group_overlap["group_incident"] == ["inc-1"]
    assert audit.conflicts, "a broken guarantee is a conflict"
    assert audit.as_dict()["clean"] is False
    corrected, meta = corrected_split(frame, "split_temporal", audit)
    assert meta["needed"] is True
    assert meta["new_split_name"] == "split_temporal_corrected"
    assignment = {r.tx_key: r.partition for r in corrected.itertuples(index=False)}
    assert assignment == {"1:0xa": TEST, "1:0xb": TEST}, "the whole group moves to test"


def test_duplicate_hash_inside_a_split_is_a_conflict():
    rows = [{"tx_key": "1:0xa", "tx_hash": "0xa", "label": 1, "block_timestamp": None,
             "split_temporal": TRAIN, "group_incident": "i", "group_protocol": None,
             "group_address": "0x1"}] * 2
    audit = audit_split(pd.DataFrame(rows), "split_temporal")
    assert audit.duplicate_hashes == ["0xa"]
    assert audit.conflicts


def test_overlap_a_split_does_not_promise_is_by_design_not_a_conflict():
    rows = [
        {"tx_key": "1:0xa", "tx_hash": "0xa", "label": 1, "block_timestamp": "2024-01-01T00:00:00Z",
         "split_temporal": TRAIN, "group_incident": "inc-1", "group_protocol": "Uniswap",
         "group_address": "0x1"},
        {"tx_key": "1:0xb", "tx_hash": "0xb", "label": 0, "block_timestamp": "2025-02-01T00:00:00Z",
         "split_temporal": TEST, "group_incident": "inc-2", "group_protocol": "Uniswap",
         "group_address": "0x1"},
    ]
    audit = audit_split(pd.DataFrame(rows), "split_temporal")
    assert audit.group_overlap["group_protocol"] == ["Uniswap"]
    assert audit.conflicts == [], "a temporal split never promised to hold protocols apart"
    assert audit.as_dict()["clean"] is True
    assert audit.as_dict()["overlap_by_design"]["group_protocol"] == 1


def test_temporal_chronology_violation_is_a_conflict():
    rows = [
        {"tx_key": "1:0xa", "tx_hash": "0xa", "label": 1, "block_timestamp": "2026-01-01T00:00:00Z",
         "split_temporal": TRAIN, "group_incident": "i1", "group_protocol": None,
         "group_address": "0x1"},
        {"tx_key": "1:0xb", "tx_hash": "0xb", "label": 0, "block_timestamp": "2025-01-01T00:00:00Z",
         "split_temporal": TEST, "group_incident": "i2", "group_protocol": None,
         "group_address": "0x2"},
    ]
    audit = audit_split(pd.DataFrame(rows), "split_temporal")
    assert audit.chronology["train_entirely_before_test"] is False
    assert any("temporal split" in c for c in audit.conflicts)
