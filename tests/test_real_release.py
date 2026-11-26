"""Checks that need the real release. Skipped automatically when it is absent.

These are the acceptance checks that cannot be done on fixtures: a manual
byte-level reconciliation against a real transaction, the survival of base data
for an incomplete record, and the task identifiers lining up across detectors.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import REPO

RELEASE = REPO / "data" / "upstream" / "txbench"
MANIFEST = REPO / "data" / "manifest" / "manifest.parquet"

pytestmark = pytest.mark.skipif(
    not (RELEASE.is_dir() and MANIFEST.is_file()),
    reason="the real release and an imported manifest are needed; run `bench fetch-release` "
           "and `bench import-txbench`")


@pytest.fixture(scope="module")
def real():
    import pandas as pd
    from txbench.release import load_release
    from txbench.upstream_status import load_trace_status
    release = load_release(RELEASE, verify_checksums=False)
    return {"release": release, "manifest": pd.read_parquet(MANIFEST),
            "traces_dir": release.path("traces_dir"),
            "status": load_trace_status(release)}


def test_real_cohort_matches_the_published_cross_check(real):
    manifest = real["manifest"]
    assert len(manifest) == 1632
    assert int((manifest["label"] == 1).sum()) == 544
    assert int((manifest["label"] == 0).sum()) == 1088
    assert int((manifest["attack_class"] == "exploit").sum()) == 386
    assert int((manifest["attack_class"] == "key_compromise").sum()) == 158
    assert set(manifest["chain_id"]) == {1}


def test_selectors_and_topics_reconcile_against_the_raw_bytes(real):
    """Manual reconciliation on one complete transaction.

    The extracted tokens are compared with the raw calldata and raw log topics
    read straight out of the release's trace file, byte by byte.
    """
    from txbench.assemble import normalized_for
    from txbench.extract import full_calls_receipt_tokens, receipt_topics, top_level_selector

    # A transaction with a call tree and several logs.
    candidates = [row.tx_hash for row in real["manifest"].itertuples(index=False)]
    chosen = None
    for tx_hash in candidates:
        raw = json.loads((real["traces_dir"] / f"{tx_hash}.json").read_text(encoding="utf-8"))
        if len(raw.get("logs") or []) >= 3 and len(raw.get("internal_transactions") or []) >= 3 \
                and (raw.get("input") or "0x") != "0x":
            chosen = (tx_hash, raw)
            break
    assert chosen, "the cohort must contain a transaction with logs and internal calls"
    tx_hash, raw = chosen

    tx = normalized_for(f"1:{tx_hash}", 1, tx_hash, traces_dir=real["traces_dir"], store=None,
                        trace_status=(real["status"].get(tx_hash) or {}).get("status"))

    # 1. the top-level selector is literally the first four bytes of `input`
    expected_selector = raw["input"][:10].lower()
    assert top_level_selector(tx).selector == expected_selector
    assert len(expected_selector) == 10

    # 2. every receipt log's first topic matches `topic0` in the raw record,
    #    in log order, and no event signature is invented
    observed = receipt_topics(tx)
    raw_logs = sorted(raw["logs"], key=lambda l: int(l["log_index"]))
    assert observed is not None and len(observed) == len(raw_logs)
    for got, want in zip(observed, raw_logs):
        assert got.raw_topic0 == (want["topic0"] or "").lower() or (
            got.raw_topic0 is None and not want["topic0"])
        assert got.log_index == int(want["log_index"])

    # 3. selector counts over all frames equal a hand count from the raw bytes
    counts, status = full_calls_receipt_tokens(tx)
    hand = {}
    frames = [raw["input"]] + [f.get("input") for f in raw["internal_transactions"]]
    precompiles = {int(f.get("to") or "0x0", 16) for f in raw["internal_transactions"]}
    for i, payload in enumerate(frames):
        if not payload or len(payload) < 10:
            continue
        if i > 0:
            target = raw["internal_transactions"][i - 1].get("to") or "0x0"
            if 0 < int(target, 16) <= 0x11:
                continue                       # a precompile keeps its own namespace
        hand[f"sel:{payload[:10].lower()}"] = hand.get(f"sel:{payload[:10].lower()}", 0) + 1
    for token, number in hand.items():
        assert counts.get(token) == number, f"{token} in {tx_hash}"
    assert status["root_frames"] == 1, "the outer call is counted exactly once"

    # record the reconciliation so the report can point at it
    out = REPO / "reports" / "reconciliation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tx_hash": tx_hash,
        "raw_input_first_4_bytes": expected_selector,
        "extracted_top_level_selector": top_level_selector(tx).selector,
        "raw_log_count": len(raw_logs),
        "extracted_log_count": len(observed),
        "raw_topic0_list": [(l["topic0"] or None) for l in raw_logs],
        "extracted_raw_topic0_list": [o.raw_topic0 for o in observed],
        "hand_counted_selector_tokens": hand,
        "extracted_selector_tokens": {k: v for k, v in counts.items() if k.startswith("sel:")},
        "frames": status["frames"], "root_frames": status["root_frames"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_an_incomplete_record_blocks_only_its_adapter_and_keeps_base_data(real):
    """A transaction whose flow graph cannot be completed still yields features."""
    import pandas as pd
    from txbench.adapters import audit
    from txbench.assemble import normalized_for
    from txbench.extract import tx_receipt_tokens

    table = REPO / "data" / "adapters" / "cohort" / "compatibility_table.parquet"
    if not table.is_file():
        pytest.skip("run `bench validate-adapters` first")
    frame = pd.read_parquet(table)
    blocked = frame[(frame.detector == "defiguard") & (frame.verdict == "blocked")]
    assert len(blocked) > 0, "the real cohort contains records with unresolvable revert ancestry"
    tx_key = blocked.iloc[0]["tx_key"]
    tx_hash = tx_key.split(":", 1)[1]

    tx = normalized_for(tx_key, 1, tx_hash, traces_dir=real["traces_dir"], store=None,
                        trace_status=(real["status"].get(tx_hash) or {}).get("status"))
    assert tx is not None, "the base record survives"
    counts, status = tx_receipt_tokens(tx)
    assert status["usable"] is True, "the tx-receipt view is unaffected"

    checks = {c.detector: c for c in audit([(tx_key, tx)])}
    assert checks["defiguard"].verdict == "blocked"
    assert checks["ft_transformer"].verdict in ("usable", "partial")
    assert "native_flows" in checks["defiguard"].missing_fields


def test_price_manipulation_ids_match_across_eligible_detectors(real):
    import pandas as pd
    tasks_file = REPO / "data" / "evaluation" / "tasks" / "tasks.json"
    task_file = REPO / "data" / "evaluation" / "tasks" / "task_price_manipulation.parquet"
    if not tasks_file.is_file():
        pytest.skip("run `bench build-tasks` first")
    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    pm = tasks["price_manipulation"]
    assert set(pm["detectors"]) == {"blockgpt", "blockscan", "blocklens", "defiguard",
                                    "ft_transformer"}, "five detectors share this task"
    frame = pd.read_parquet(task_file)
    positives = set(frame.loc[frame.role == "positive", "tx_key"])
    controls = set(frame.loc[frame.role == "control", "tx_key"])
    ambiguous = set(frame.loc[frame.role == "ambiguous_unresolved", "tx_key"])
    assert positives and controls
    assert not (positives & controls) and not (positives & ambiguous)
    assert len(positives) == pm["positives"] and len(controls) == pm["negatives"]
    # the same identifiers are what every eligible detector is scored on
    assert len(ambiguous) == pm["ambiguous_left_unresolved"] > 0, \
        "an ambiguous mapping is left unresolved rather than forced into a class"


def test_every_real_transaction_has_a_complete_call_tree_per_the_release(real):
    status = real["status"]
    hashes = list(real["manifest"]["tx_hash"])
    assessed = [status[h]["status"] for h in hashes if h in status]
    assert len(assessed) == len(hashes), "the release assesses every cohort transaction"
    assert set(assessed) == {"ok"}


def test_no_real_transaction_supplies_call_nesting(real):
    """The binding gap, asserted on real data: frames carry no parent pointers."""
    from txbench.assemble import normalized_for
    sample = list(real["manifest"]["tx_hash"])[:40]
    for tx_hash in sample:
        tx = normalized_for(f"1:{tx_hash}", 1, tx_hash, traces_dir=real["traces_dir"],
                            store=None,
                            trace_status=(real["status"].get(tx_hash) or {}).get("status"))
        if tx is None or len(tx.frames) <= 1:
            continue
        assert tx.call_tree_has_nesting is False
        assert "call_tree_nesting" in tx.field_gaps


def test_the_recorded_provenance_names_the_release_doi():
    """Section 1 requires the DOI on the published manifest's provenance."""
    provenance = json.loads((REPO / "data" / "manifest" / "release_provenance.json").read_text())
    assert provenance["doi"], "release_provenance.json records no DOI"
    assert provenance["release_id"] == "TxBench-v1.0.1"
