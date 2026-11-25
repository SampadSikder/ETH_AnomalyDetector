"""Planning, pilot selection and resume behaviour - all offline."""
from __future__ import annotations

import pytest

from txbench.cache import CacheRef, CacheStore
from txbench.collect import build_plan, choose_pilot, estimate_cost, execute_plan, make_budget
from txbench.config import M_RECEIPT, M_TX, load_config
from txbench.rpc import build_providers


def test_pilot_selection_is_deterministic(manifest):
    first = choose_pilot(manifest, 6, seed=42)
    second = choose_pilot(manifest, 6, seed=42)
    assert first == second
    assert len(first) == min(6, len(manifest))


def test_pilot_covers_both_classes(manifest):
    labels = {row.tx_key: int(row.label) for row in manifest.itertuples(index=False)}
    picked = choose_pilot(manifest, 6, seed=42)
    assert {labels[k] for k in picked} == {0, 1}


def test_pilot_spans_several_periods(manifest):
    stamps = {row.tx_key: str(row.block_timestamp)[:4]
              for row in manifest.itertuples(index=False)}
    picked = choose_pilot(manifest, 8, seed=42)
    assert len({stamps[k] for k in picked}) > 1


def test_plan_needs_no_calls_when_upstream_supplies_everything(manifest, traces_dir,
                                                               trace_status, tmp_path):
    cfg = load_config()
    store = CacheStore(tmp_path)
    status = {f"1:{h}": (r.get("status") or "") for h, r in trace_status.items()}
    plan = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                      trace_status=status, provider_order=["alchemy"])
    summary = plan.as_dict()
    # The fixture includes one transaction with no trace at all, which is the
    # only thing that needs fetching.
    assert summary["upstream_supplied"]["trace_missing"] == 1
    assert summary["planned_calls_by_method"] == {M_TX: 1, M_RECEIPT: 1}


def test_a_cached_request_is_not_planned_again(manifest, traces_dir, trace_status, tmp_path):
    cfg = load_config()
    store = CacheStore(tmp_path)
    status = {f"1:{h}": (r.get("status") or "") for h, r in trace_status.items()}
    plan = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                      trace_status=status, provider_order=["alchemy"])
    assert plan.planned, "the fixture has one transaction that needs fetching"
    for call in plan.planned:
        store.put(CacheRef.for_tx(1, call.target_id, call.method), {"hash": call.target_id},
                  provider="fake")
    again = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                       trace_status=status, provider_order=["alchemy"])
    assert again.planned == [], "a completed request is never re-requested"
    assert sum(again.cache_hits.values()) == 2


def test_a_permanently_unsupported_request_is_not_retried(manifest, traces_dir, trace_status,
                                                          tmp_path):
    from txbench.cache import UNSUPPORTED
    cfg = load_config()
    store = CacheStore(tmp_path)
    status = {f"1:{h}": (r.get("status") or "") for h, r in trace_status.items()}
    plan = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                      trace_status=status, provider_order=["alchemy"])
    for call in plan.planned:
        store.record_failure(CacheRef.for_tx(1, call.target_id, call.method),
                             state=UNSUPPORTED, provider="fake", error="not supported")
    again = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                       trace_status=status, provider_order=["alchemy"])
    assert again.planned == []
    assert sum(again.cache_permanent_misses.values()) == 2


def test_a_retryable_failure_is_planned_again(manifest, traces_dir, trace_status, tmp_path):
    from txbench.cache import FAILED_RETRYABLE
    cfg = load_config()
    store = CacheStore(tmp_path)
    status = {f"1:{h}": (r.get("status") or "") for h, r in trace_status.items()}
    plan = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                      trace_status=status, provider_order=["alchemy"])
    for call in plan.planned:
        store.record_failure(CacheRef.for_tx(1, call.target_id, call.method),
                             state=FAILED_RETRYABLE, provider="fake", error="timeout")
    again = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                       trace_status=status, provider_order=["alchemy"])
    assert len(again.planned) == len(plan.planned), "a timeout is worth another attempt"


def test_dry_run_prices_the_plan_without_a_credential(manifest, traces_dir, trace_status,
                                                      tmp_path):
    cfg = load_config()
    cfg.collect.verify_upstream = True
    store = CacheStore(tmp_path)
    status = {f"1:{h}": (r.get("status") or "") for h, r in trace_status.items()}
    plan = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                      trace_status=status, provider_order=["alchemy", "drpc"])
    estimate = plan.estimates["providers"]["alchemy"]
    assert estimate["credential_present"] is False
    assert estimate["usable_now"] is False
    assert estimate["calls_total"] == len(plan.planned) > 0
    assert estimate["estimated_units"] == estimate["calls_total"] * 20
    assert estimate["within_run_budget"] is True


def test_trace_methods_are_never_priced_or_planned(manifest, traces_dir, trace_status, tmp_path):
    cfg = load_config()
    store = CacheStore(tmp_path)
    status = {f"1:{h}": (r.get("status") or "") for h, r in trace_status.items()}
    plan = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                      trace_status=status, provider_order=["alchemy", "drpc"])
    for provider, estimate in plan.estimates["providers"].items():
        assert "debug_traceTransaction" not in estimate["calls_by_method"]
        assert "debug_traceTransaction" in estimate["disabled_methods"]


def test_execute_without_a_provider_stops_cleanly(manifest, traces_dir, trace_status, tmp_path):
    cfg = load_config()
    store = CacheStore(tmp_path)
    status = {f"1:{h}": (r.get("status") or "") for h, r in trace_status.items()}
    plan = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                      trace_status=status, provider_order=["alchemy"])
    budget = make_budget(cfg.collect, cfg.providers)
    providers = build_providers(cfg.providers, budget=budget)
    outcome = execute_plan(plan, cfg=cfg, store=store, providers=providers, budget=budget)
    assert outcome.calls_made == 0
    assert "no credential present" in outcome.stopped_early


def test_dry_run_makes_no_calls(manifest, traces_dir, trace_status, tmp_path):
    cfg = load_config()
    store = CacheStore(tmp_path)
    status = {f"1:{h}": (r.get("status") or "") for h, r in trace_status.items()}
    plan = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                      trace_status=status, provider_order=["alchemy"])
    budget = make_budget(cfg.collect, cfg.providers)
    providers = build_providers(cfg.providers, budget=budget)
    outcome = execute_plan(plan, cfg=cfg, store=store, providers=providers, budget=budget,
                           dry_run=True)
    assert outcome.calls_made == 0 and outcome.failures == []
    assert list(store.iter_entries()) == []
