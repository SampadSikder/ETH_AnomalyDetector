"""The cache-first collector: inventory, dry run, pilot, resumable full run.

Order of operations, deliberately:
  1. read what the upstream release already supplies (`inventory`);
  2. consult the cache;
  3. only then plan provider calls, and price the plan *before* making it;
  4. fetch with per-provider limits, bounded retries and a budget fuse;
  5. reconcile every fetched record against the upstream one and store it with
     provenance.

A resumed run reuses every completed request and preserves all manifest rows.
A second completed run makes zero data calls.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .cache import (CacheRef, CacheStore, FAILED_RETRYABLE, INVALID, OK, TARGET_BLOCK,
                    TARGET_TX, UNSUPPORTED, utc_now)
from .config import (BASE_METHODS, CollectConfig, Config, M_BLOCK_BY_NUMBER, M_CHAIN_ID,
                     M_RECEIPT, M_TX)
from .errors import BudgetExceeded, ProviderError
from .ids import normalize_tx_hash
from .inventory import TxInventory, normalized_from_upstream, take_inventory
from .logs import get_logger
from .rpc import Budget, Provider, build_providers
from .credentials import credential_status, redact

log = get_logger("collect")


# --- pilot selection ----------------------------------------------------------

def choose_pilot(manifest: pd.DataFrame, size: int, seed: int,
                 trace_sizes: dict[str, int] | None = None) -> list[str]:
    """A deterministic pilot covering both classes, several periods and both
    short and long traces where the release supplies sizes.

    Determinism: strata are built from sorted keys and sampled with a seeded
    RNG, so the same manifest, size and seed always give the same hashes.
    """
    if not len(manifest):
        return []
    rows = manifest.sort_values("tx_key")
    sizes = trace_sizes or {}
    size_values = sorted(v for v in sizes.values() if v)
    if size_values:
        lo = size_values[len(size_values) // 3]
        hi = size_values[(2 * len(size_values)) // 3]
    else:
        lo = hi = None

    def size_band(key: str) -> str:
        value = sizes.get(key)
        if value is None or lo is None:
            return "unknown"
        return "short" if value <= lo else ("long" if value >= hi else "mid")

    def period(stamp: object) -> str:
        text = str(stamp or "")
        return text[:4] if len(text) >= 4 and text[:4].isdigit() else "unknown"

    strata: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for row in rows.itertuples(index=False):
        key = row.tx_key
        strata[(int(row.label), period(getattr(row, "block_timestamp", None)),
                size_band(key))].append(key)

    rng = random.Random(seed)
    picked: list[str] = []
    ordered = sorted(strata.items(), key=lambda kv: (str(kv[0]), -len(kv[1])))
    # Round-robin over strata so both classes, several years and both trace
    # lengths appear before any stratum is sampled twice.
    pools = {k: list(v) for k, v in ordered}
    for pool in pools.values():
        rng.shuffle(pool)
    while len(picked) < size and any(pools.values()):
        for key in list(pools):
            if len(picked) >= size:
                break
            if pools[key]:
                picked.append(pools[key].pop())
    return sorted(picked)


# --- planning -----------------------------------------------------------------

@dataclass
class PlannedCall:
    method: str
    target_kind: str
    target_id: str
    tx_keys: list[str] = field(default_factory=list)

    @property
    def ref_args(self) -> tuple[str, str, str]:
        return self.method, self.target_kind, self.target_id


@dataclass
class CollectionPlan:
    chain_id: int
    inventories: list[TxInventory]
    planned: list[PlannedCall]
    cache_hits: dict[str, int]
    cache_permanent_misses: dict[str, int]
    unique_blocks: int
    provider_order: list[str]
    estimates: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        by_method = Counter(p.method for p in self.planned)
        return {
            "chain_id": self.chain_id,
            "transactions": len(self.inventories),
            "planned_calls_total": len(self.planned),
            "planned_calls_by_method": dict(sorted(by_method.items())),
            "unique_blocks_needed": self.unique_blocks,
            "cache_hits_by_method": dict(sorted(self.cache_hits.items())),
            "cache_permanent_misses_by_method": dict(sorted(self.cache_permanent_misses.items())),
            "upstream_supplied": {
                "envelope_complete": sum(i.envelope_complete for i in self.inventories),
                "receipt_present": sum(i.receipt_present for i in self.inventories),
                "call_tree_complete": sum(i.call_tree_status == "complete"
                                          for i in self.inventories),
                "call_tree_with_nesting": sum(i.call_tree_has_nesting for i in self.inventories),
                "trace_missing": sum(not i.upstream_present for i in self.inventories),
            },
            "provider_order": self.provider_order,
            "estimates": self.estimates,
        }


def build_plan(manifest: pd.DataFrame, *, cfg: Config, store: CacheStore,
               traces_dir: Path | None, trace_status: dict[str, str] | None = None,
               provider_order: Sequence[str] = ("alchemy", "drpc"),
               providers: dict[str, Provider] | None = None,
               only_keys: Iterable[str] | None = None) -> CollectionPlan:
    """Inventory every row, consult the cache, and price what is left."""
    keys = set(only_keys) if only_keys is not None else None
    rows = manifest if keys is None else manifest[manifest["tx_key"].isin(keys)]
    statuses = trace_status or {}

    inventories: list[TxInventory] = []
    for row in rows.sort_values("tx_key").itertuples(index=False):
        tx_hash = normalize_tx_hash(row.tx_hash)
        tx, size = (None, None)
        if traces_dir is not None:
            tx, size = normalized_from_upstream(traces_dir, int(row.chain_id), tx_hash,
                                                statuses.get(row.tx_key))
        inventories.append(take_inventory(tx, chain_id=int(row.chain_id), tx_hash=tx_hash,
                                          traces_dir=traces_dir, payload_bytes=size,
                                          verify_upstream=cfg.collect.verify_upstream))

    wanted: dict[tuple[str, str, str], PlannedCall] = {}
    hits: Counter[str] = Counter()
    permanent: Counter[str] = Counter()
    for inv in inventories:
        for method, kind, target in inv.needs:
            ref = _ref(cfg.chain_id, method, kind, target)
            entry = store.get(ref)
            if entry and entry.is_hit:
                hits[method] += 1
                continue
            if entry and entry.is_permanent_miss:
                permanent[method] += 1
                continue
            call = wanted.setdefault((method, kind, target),
                                     PlannedCall(method=method, target_kind=kind, target_id=target))
            call.tx_keys.append(inv.tx_key)

    planned = sorted(wanted.values(), key=lambda c: (c.method, c.target_kind, c.target_id))
    unique_blocks = len({c.target_id for c in planned if c.target_kind == TARGET_BLOCK})
    order = [p for p in provider_order if p in cfg.providers]
    plan = CollectionPlan(chain_id=cfg.chain_id, inventories=inventories, planned=planned,
                          cache_hits=dict(hits), cache_permanent_misses=dict(permanent),
                          unique_blocks=unique_blocks, provider_order=order)
    plan.estimates = estimate_cost(plan, cfg=cfg, providers=providers)
    return plan


def estimate_cost(plan: CollectionPlan, *, cfg: Config,
                  providers: dict[str, Provider] | None = None) -> dict[str, Any]:
    """Per-provider, per-method estimate for the calls the plan would make."""
    out: dict[str, Any] = {"note": "estimate for the calls this plan would make; cache hits "
                                   "and permanently unsupported methods are excluded",
                           "providers": {}}
    remaining = Counter(p.method for p in plan.planned)
    for name in plan.provider_order:
        pcfg = cfg.provider(name)
        provider = providers.get(name) if providers else None
        serves: dict[str, int] = {}
        units = 0
        for method, count in sorted(remaining.items()):
            # Price by what the *plan* allows, not by what is reachable right
            # now: a dry run must cost the work even with no credential set.
            if method in set(pcfg.disabled_methods):
                continue
            if provider is not None and method in provider._disabled_at_runtime:
                continue
            serves[method] = count
            units += count * pcfg.unit_cost(method)
        out["providers"][name] = {
            "unit": pcfg.unit,
            "unit_cost_source": pcfg.units_source,
            "calls_by_method": serves,
            "calls_total": sum(serves.values()),
            "estimated_units": units,
            "run_unit_budget": pcfg.run_unit_budget,
            "within_run_budget": units <= pcfg.run_unit_budget,
            "plan_unit_allowance": pcfg.plan_unit_allowance,
            "share_of_plan_allowance": (round(units / pcfg.plan_unit_allowance, 6)
                                        if pcfg.plan_unit_allowance else None),
            "credential_present": (provider.has_credential if provider
                                   else bool(credential_status().get(pcfg.key_var))),
            "usable_now": bool(provider.available) if provider else False,
            "pricing_basis": "configured method costs applied to the planned calls; a missing "
                             "credential does not change the price, only whether the run can "
                             "proceed",
            "disabled_methods": sorted(set(pcfg.disabled_methods)),
            "concurrency": pcfg.concurrency,
            "rate_limit_units_per_s": pcfg.rate_limit_units_per_s,
        }
        # The primary serves everything it can; the fallback is priced for the
        # same calls, since it is only used when the primary fails.
    out["calls_total"] = len(plan.planned)
    out["calls_by_method"] = dict(sorted(remaining.items()))
    return out


def _ref(chain_id: int, method: str, kind: str, target: str) -> CacheRef:
    if kind == TARGET_BLOCK:
        return CacheRef.for_block(chain_id, int(target), method)
    return CacheRef.for_tx(chain_id, target, method)


# --- capability probe ---------------------------------------------------------

def probe_providers(providers: dict[str, Provider], *, chain_id: int,
                    sample_hashes: Sequence[str], methods: Sequence[str] = BASE_METHODS,
                    store: CacheStore | None = None) -> dict[str, Any]:
    """Assess each provider on the same small set of historical hashes.

    Cheap, bounded, and run before a provider is trusted with a full pass.
    """
    report: dict[str, Any] = {"chain_id": chain_id, "sample_hashes": list(sample_hashes),
                              "providers": {}}
    for name, provider in providers.items():
        entry: dict[str, Any] = {**provider.status(), "methods": {}, "chain_id_check": None}
        if not provider.available:
            entry["skipped"] = ("no credential for " + provider.cfg.key_var
                                if not provider.has_credential else "disabled")
            report["providers"][name] = entry
            continue
        try:
            outcome = provider.call(M_CHAIN_ID, [])
            observed = provider.result_of(outcome)
            observed_id = int(str(observed), 16) if isinstance(observed, str) else int(observed)
            entry["chain_id_check"] = {"observed": observed_id, "expected": chain_id,
                                       "ok": observed_id == chain_id}
            if observed_id != chain_id:
                entry["skipped"] = f"endpoint reports chain {observed_id}, expected {chain_id}"
                report["providers"][name] = entry
                continue
        except ProviderError as exc:
            entry["chain_id_check"] = {"error": redact(exc)}
        for method in methods:
            if not provider.supports(method):
                entry["methods"][method] = {"supported": False, "reason": "disabled by config"}
                continue
            result: dict[str, Any] = {"supported": None, "attempts": 0, "ok": 0, "errors": []}
            for tx_hash in sample_hashes[:2]:
                params = _params_for(method, tx_hash)
                if params is None:
                    continue
                result["attempts"] += 1
                try:
                    outcome = provider.call(method, params)
                    payload = provider.result_of(outcome)
                    result["ok"] += int(payload is not None)
                    result["supported"] = True
                except ProviderError as exc:
                    result["errors"].append(redact(exc)[:200])
                    if exc.unsupported:
                        result["supported"] = False
                        break
                except BudgetExceeded as exc:
                    result["errors"].append(redact(exc)[:200])
                    break
            entry["methods"][method] = result
        report["providers"][name] = entry
    return report


def _params_for(method: str, tx_hash: str) -> list[Any] | None:
    if method in (M_TX, M_RECEIPT):
        return [tx_hash]
    return None


# --- execution ----------------------------------------------------------------

@dataclass
class CollectionOutcome:
    calls_made: int = 0
    cache_hits: int = 0
    stored_payload_bytes: int = 0
    by_method: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    failures: list[dict[str, Any]] = field(default_factory=list)
    reconciliation: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    stopped_early: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls_made": self.calls_made,
            "cache_hits_skipped": self.cache_hits,
            "stored_payload_bytes": self.stored_payload_bytes,
            "by_method": {m: dict(c) for m, c in sorted(self.by_method.items())},
            "failures": self.failures[:200],
            "failure_count": len(self.failures),
            "reconciliation_mismatches": [r for r in self.reconciliation if r["mismatches"]],
            "reconciliation_checked": len(self.reconciliation),
            "budget": self.budget,
            "stopped_early": self.stopped_early,
        }


def execute_plan(plan: CollectionPlan, *, cfg: Config, store: CacheStore,
                 providers: dict[str, Provider], budget: Budget,
                 upstream: dict[str, Any] | None = None,
                 dry_run: bool = False) -> CollectionOutcome:
    """Make the planned calls, cheapest-provider-first, and store with provenance."""
    outcome = CollectionOutcome()
    outcome.budget = budget.snapshot()
    if dry_run:
        return outcome
    if not plan.planned:
        outcome.stopped_early = ("nothing to fetch: every field these transactions need is "
                                 "already supplied by the upstream release or the cache")
        log.info(outcome.stopped_early)
        return outcome

    order = [providers[n] for n in plan.provider_order if n in providers]
    usable = [p for p in order if p.available]
    if not usable:
        outcome.stopped_early = ("no provider is usable: no credential present for any "
                                 "configured provider")
        log.warning(outcome.stopped_early)
        return outcome

    max_workers = max(1, max(p.cfg.concurrency for p in usable))
    lock_free_results: list[tuple[PlannedCall, str | None, Any, bytes | None, int, str | None]] = []

    def one(call: PlannedCall):
        last_error: str | None = None
        for provider in usable:
            if not provider.supports(call.method):
                continue
            units = provider.unit_cost(call.method)
            if budget.would_exceed(provider.name, units):
                last_error = f"{provider.name}: budget fuse would trip"
                continue
            try:
                params = _call_params(call)
                result = provider.call(call.method, params)
                payload = provider.result_of(result)
                return call, provider.name, payload, result.raw_bytes, result.units, None
            except ProviderError as exc:
                last_error = redact(exc)
                if exc.unsupported:
                    continue            # try the next provider for this method
                continue
            except BudgetExceeded as exc:
                last_error = redact(exc)
                break
        return call, None, None, None, 0, last_error or "no provider served the call"

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for item in pool.map(one, plan.planned):
                lock_free_results.append(item)
    except BudgetExceeded as exc:
        outcome.stopped_early = redact(exc)

    upstream_index = upstream or {}
    for call, provider_name, payload, raw, units, error in lock_free_results:
        ref = _ref(plan.chain_id, *call.ref_args[0:1], *call.ref_args[1:])
        if provider_name is None or payload is None:
            state = UNSUPPORTED if error and "unsupported" in error.lower() else FAILED_RETRYABLE
            store.record_failure(ref, state=state, provider=provider_name, error=error or "unknown")
            outcome.by_method[call.method][state] += 1
            outcome.failures.append({"method": call.method, "target": call.target_id,
                                     "state": state, "error": error})
            continue
        checks = _validate_payload(call, payload, plan.chain_id)
        if checks["invalid_reason"]:
            store.record_failure(ref, state=INVALID, provider=provider_name,
                                 error=checks["invalid_reason"])
            outcome.by_method[call.method][INVALID] += 1
            outcome.failures.append({"method": call.method, "target": call.target_id,
                                     "state": INVALID, "error": checks["invalid_reason"]})
            continue
        entry = store.put(ref, payload, provider=provider_name, raw_bytes=raw,
                          completeness=checks["completeness"],
                          block_number=checks["block_number"], block_hash=checks["block_hash"],
                          tx_index=checks["tx_index"], units_charged=units,
                          max_payload_bytes=cfg.collect.max_stored_payload_bytes)
        outcome.calls_made += 1
        outcome.by_method[call.method][entry.state] += 1
        outcome.stored_payload_bytes += entry.payload_bytes or 0
        for tx_key in call.tx_keys:
            reference = upstream_index.get(tx_key)
            if reference is not None:
                outcome.reconciliation.append(
                    _reconcile(tx_key, call.method, payload, reference))
    outcome.budget = budget.snapshot()
    return outcome


def _call_params(call: PlannedCall) -> list[Any]:
    if call.method in (M_TX, M_RECEIPT):
        return [call.target_id]
    if call.method == M_BLOCK_BY_NUMBER:
        return [hex(int(call.target_id)), False]
    raise ProviderError(f"no parameter builder for {call.method}")


def _validate_payload(call: PlannedCall, payload: Any, chain_id: int) -> dict[str, Any]:
    """Cross-check identity before anything is stored."""
    out = {"invalid_reason": None, "completeness": "complete", "block_number": None,
           "block_hash": None, "tx_index": None}
    if not isinstance(payload, dict):
        out["invalid_reason"] = f"{call.method}: reply is {type(payload).__name__}, not an object"
        return out
    if call.target_kind == TARGET_TX:
        got = str(payload.get("transactionHash") or payload.get("hash") or "").lower()
        if got and got != call.target_id:
            out["invalid_reason"] = f"{call.method}: reply is for a different transaction"
            return out
        out["block_number"] = _hex_int(payload.get("blockNumber"))
        out["block_hash"] = _lower(payload.get("blockHash"))
        out["tx_index"] = _hex_int(payload.get("transactionIndex"))
        if call.method == M_RECEIPT and "logs" not in payload:
            out["completeness"] = "partial_no_logs"
    else:
        out["block_number"] = _hex_int(payload.get("number"))
        out["block_hash"] = _lower(payload.get("hash"))
        if out["block_number"] is not None and out["block_number"] != int(call.target_id):
            out["invalid_reason"] = f"{call.method}: reply is for block {out['block_number']}"
    return out


def _reconcile(tx_key: str, method: str, payload: dict[str, Any],
               reference: dict[str, Any]) -> dict[str, Any]:
    """Compare a fetched record against the upstream one, field by field."""
    mismatches: dict[str, Any] = {}
    pairs = {
        "block_number": (_hex_int(payload.get("blockNumber")), reference.get("block_number")),
        "block_hash": (_lower(payload.get("blockHash")), reference.get("block_hash")),
        "transaction_index": (_hex_int(payload.get("transactionIndex")),
                              reference.get("transaction_index")),
    }
    if method == M_TX:
        pairs["input"] = (_lower(payload.get("input")), reference.get("input_hex"))
        pairs["from"] = (_lower(payload.get("from")), reference.get("from_address"))
        pairs["to"] = (_lower(payload.get("to")), reference.get("to_address"))
    if method == M_RECEIPT:
        pairs["log_count"] = (len(payload.get("logs") or []) if "logs" in payload else None,
                              reference.get("log_count"))
        pairs["status"] = (_hex_int(payload.get("status")), reference.get("receipt_status"))
    for name, (fetched, upstream_value) in pairs.items():
        if fetched is None or upstream_value is None:
            continue
        if fetched != upstream_value:
            mismatches[name] = {"fetched": fetched, "upstream": upstream_value}
    return {"tx_key": tx_key, "method": method, "mismatches": mismatches,
            "checked": sorted(k for k, (a, b) in pairs.items() if a is not None and b is not None)}


def _hex_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(text, 10)
    except ValueError:
        return None


def _lower(value: Any) -> str | None:
    return str(value).strip().lower() if value is not None else None


def write_inventory(plan: CollectionPlan, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([i.as_row() for i in plan.inventories])
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def make_budget(cfg: CollectConfig, providers_cfg: dict[str, Any]) -> Budget:
    return Budget(max_calls=cfg.max_calls,
                  per_provider_units={n: p.run_unit_budget for n, p in providers_cfg.items()})
