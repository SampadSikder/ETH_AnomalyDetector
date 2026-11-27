"""`bench` - the project's command line.

    bench import-txbench --release-dir data/upstream/txbench --chain-id 1
    bench collect --dry-run
    bench collect --pilot 20 --seed 42 --provider alchemy --fallback drpc
    bench audit-inputs --pilot
    bench collect --resume --provider alchemy --fallback drpc
    bench build-features --view tx-receipt
    bench build-features --view full-calls-receipt
    bench verify-splits
    bench validate-adapters
    bench evaluate --predictions fixtures/example_predictions.csv --fixture-mode
    bench report
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .credentials import credential_status, load_dotenv, redact
from .errors import TxBenchError
from .logs import get_logger, setup_logging
from .paths import layout as make_layout

log = get_logger("cli")


# --------------------------------------------------------------------------- #
# shared setup
# --------------------------------------------------------------------------- #

def _context(args: argparse.Namespace):
    from .config import load_config
    layout = make_layout(Path(args.root) if getattr(args, "root", None) else None)
    load_dotenv(layout.root)
    cfg = load_config(Path(args.config) if getattr(args, "config", None) else None,
                      root=layout.root)
    if getattr(args, "chain_id", None):
        cfg.chain_id = int(args.chain_id)
    if getattr(args, "release_dir", None):
        cfg.release_dir = Path(args.release_dir)
    layout.ensure()
    return cfg, layout


def _release(cfg, layout, args: argparse.Namespace):
    from .release import load_release
    directory = Path(args.release_dir) if getattr(args, "release_dir", None) else cfg.release_dir
    if not directory.is_absolute():
        directory = layout.root / directory
    # An explicit --doi wins; otherwise the DOI pinned in the config is used, so
    # a plain re-import still records it instead of silently dropping it.
    return load_release(directory, release_id=getattr(args, "release_id", None),
                        doi=getattr(args, "doi", None) or cfg.release_doi,
                        verify_checksums=not getattr(args, "skip_checksums", False))


def _manifest(layout):
    from .importer import load_manifest
    if not layout.manifest.is_file():
        raise TxBenchError(f"no manifest at {layout.manifest}. Run `bench import-txbench` first.")
    return load_manifest(layout.manifest)


def _emit(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_fetch_release(args: argparse.Namespace) -> int:
    """Download, checksum and extract the pinned upstream release."""
    from .fetch import fetch_release
    cfg, layout = _context(args)
    result = fetch_release(layout, record_id=args.record, keep_archive=not args.no_keep_archive,
                           expected_md5=args.md5)
    _emit(result)
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    from .importer import import_labels, write_manifest
    from .release import write_provenance
    from .report import write_cohort_report
    cfg, layout = _context(args)
    rel = _release(cfg, layout, args)
    result = import_labels(rel, chain_id=cfg.chain_id)
    written = write_manifest(result, manifest_path=layout.manifest,
                             meta_path=layout.manifest_meta,
                             mapping_path=layout.schema_mapping,
                             quarantine_path=layout.quarantine)
    written["release_provenance"] = write_provenance(
        rel, layout.manifest_dir / "release_provenance.json")
    written["cohort_report"] = write_cohort_report(result, layout.cohort_report)
    _emit({"rows": int(len(result.manifest)),
           "label_counts": result.meta["label_counts"],
           "attack_class_counts": result.meta["attack_class_counts"],
           "quarantined_rows": result.meta["quarantined_rows"],
           "cohort_sha256": result.cohort_sha256,
           "is_fixture": result.meta["is_fixture"],
           "filters": result.meta["filters"],
           "artifacts": {k: str(v) for k, v in written.items()}})
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    from .cache import CacheStore, write_cache_index_report
    from .collect import (build_plan, choose_pilot, execute_plan, make_budget, probe_providers,
                          write_inventory, write_json)
    from .inventory import load_upstream_trace
    from .rpc import build_providers
    from .upstream_status import load_trace_status
    cfg, layout = _context(args)
    manifest = _manifest(layout)
    rel = None
    traces_dir = None
    statuses: dict[str, dict[str, Any]] = {}
    try:
        rel = _release(cfg, layout, args)
        traces_dir = rel.path("traces_dir") if rel.has("traces_dir") else None
        statuses = load_trace_status(rel)
    except TxBenchError as exc:
        log.warning("proceeding without the upstream release: %s", redact(exc))

    if args.verify_upstream:
        cfg.collect.verify_upstream = True
    store = CacheStore(layout.cache_dir, layout.cache_index)
    collection_dir = layout.data / "collection"
    # The cross-check scenario is priced into its own files so the base plan
    # (fetch only what upstream lacks) stays the one the report quotes.
    scenario = "verify_upstream" if cfg.collect.verify_upstream else "base"
    suffix = "" if scenario == "base" else f"_{scenario}"

    keys = None
    if args.pilot:
        sizes: dict[str, int] = {}
        if traces_dir is not None:
            for row in manifest.itertuples(index=False):
                _, size = load_upstream_trace(traces_dir, row.tx_hash)
                if size:
                    sizes[row.tx_key] = size
        keys = choose_pilot(manifest, args.pilot, args.seed, sizes)
        write_json({"pilot_size": args.pilot, "seed": args.seed, "keys": keys,
                    "selection": "deterministic: strata over (label, year, trace-size band), "
                                 "round-robin, seeded shuffle"},
                   collection_dir / "pilot_keys.json")
        log.info("pilot: %d transactions", len(keys))

    order = [args.provider] + ([args.fallback] if args.fallback else [])
    budget = make_budget(cfg.collect, cfg.providers)
    providers = build_providers(cfg.providers, budget=budget,
                                timeout_s=cfg.collect.request_timeout_s,
                                max_retries=cfg.collect.max_retries,
                                base_delay_s=cfg.collect.retry_base_delay_s,
                                max_delay_s=cfg.collect.retry_max_delay_s)
    status_by_key = {f"{cfg.chain_id}:{h}": (r.get("status") or "")
                     for h, r in statuses.items()}
    plan = build_plan(manifest, cfg=cfg, store=store, traces_dir=traces_dir,
                      trace_status=status_by_key, provider_order=order, providers=providers,
                      only_keys=keys)
    write_inventory(plan, collection_dir / f"inventory{suffix}.parquet")
    plan_doc = {**plan.as_dict(), "scenario": scenario,
                "scenario_meaning": (
                    "base: fetch only what the upstream release does not supply"
                    if scenario == "base" else
                    "verify_upstream: re-read the envelope and receipt for every transaction to "
                    "cross-check the release against the chain, even where the release already "
                    "supplies the field"),
                "effective_settings": cfg.effective_settings(),
                "credentials_present": credential_status(),
                "providers": {n: p.status() for n, p in providers.items()}}
    write_json(plan_doc, collection_dir / f"plan{suffix}.json")

    if args.dry_run:
        _emit({"dry_run": True, **{k: v for k, v in plan_doc.items()
                                   if k != "effective_settings"}})
        return 0

    probe = None
    if not args.no_probe:
        sample = [k.split(":", 1)[1] for k in (plan.inventories and
                                               [i.tx_key for i in plan.inventories[:2]] or [])]
        probe = probe_providers(providers, chain_id=cfg.chain_id, sample_hashes=sample,
                                store=store)
        write_json(probe, collection_dir / "provider_probe.json")

    upstream_reference = {}
    if traces_dir is not None:
        from .assemble import normalized_for
        for inv in plan.inventories:
            tx = normalized_for(inv.tx_key, inv.chain_id, inv.tx_hash, traces_dir=traces_dir,
                                store=None, trace_status=status_by_key.get(inv.tx_key))
            if tx is not None:
                upstream_reference[inv.tx_key] = {
                    "block_number": tx.block_number, "block_hash": tx.block_hash,
                    "transaction_index": tx.transaction_index, "input_hex": tx.input_hex,
                    "from_address": tx.from_address, "to_address": tx.to_address,
                    "log_count": tx.log_count, "receipt_status": tx.receipt_status}

    outcome = execute_plan(plan, cfg=cfg, store=store, providers=providers, budget=budget,
                           upstream=upstream_reference)
    report = {"plan": plan.as_dict(), "outcome": outcome.as_dict(),
              "cache": store.stats(), "provider_probe": probe,
              "effective_settings": cfg.effective_settings()}
    write_json(report, collection_dir / "collection_report.json")
    write_cache_index_report(store, collection_dir / "cache_index.parquet")
    _emit({"calls_made": outcome.calls_made, "cache_hits_skipped": plan.cache_hits,
           "failures": len(outcome.failures), "budget": outcome.budget,
           "stopped_early": outcome.stopped_early,
           "artifacts": str(collection_dir / "collection_report.json")})
    return 0


def _assembled(cfg, layout, args, keys=None):
    """Normalised records for the cohort (or a subset), upstream first.

    Call trees already recovered by `bench enrich-traces` are applied here, so
    every command downstream - the audit, the smoke test, the feature build -
    sees the records a detector would actually be handed. Offline: the trees
    come from the local cache, and a hash without one is left as the release
    supplied it.
    """
    from .assemble import records
    from .cache import CacheStore
    from .upstream_status import load_trace_status
    manifest = _manifest(layout)
    traces_dir = None
    statuses: dict[str, dict[str, Any]] = {}
    try:
        rel = _release(cfg, layout, args)
        traces_dir = rel.path("traces_dir") if rel.has("traces_dir") else None
        statuses = load_trace_status(rel)
    except TxBenchError as exc:
        log.warning("no upstream release available: %s", redact(exc))
        rel = None
    store = CacheStore(layout.cache_dir, layout.cache_index)
    assembled = list(records(manifest, traces_dir=traces_dir, store=store,
                             trace_status=statuses, keys=keys))
    _assembled.last_tree_application = None
    if not getattr(args, "ignore_recovered_traces", False):
        from .enrich import apply_cached_trees
        applied = apply_cached_trees(assembled, store=store, chain_id=cfg.chain_id,
                                     table=cfg.bigquery.table)
        _assembled.last_tree_application = applied.as_dict()
        if applied.applied:
            log.info("applied %d recovered call tree(s) from the local cache; %d complete",
                     applied.applied, applied.complete)
        if applied.root_disagreements:
            log.warning("%d recovered tree(s) were NOT applied: the root disagreed with the "
                        "local transaction", len(applied.root_disagreements))
    return manifest, rel, assembled


def _pilot_keys(layout) -> list[str] | None:
    path = layout.data / "collection" / "pilot_keys.json"
    if not path.is_file():
        return None
    return list(json.loads(path.read_text(encoding="utf-8")).get("keys") or [])


def cmd_audit_inputs(args: argparse.Namespace) -> int:
    """Per-detector field coverage on the pilot (or the whole cohort)."""
    from .adapters import audit, write_audit
    from .adapters.blockscan_smoke import default_repo_dir, run_smoke
    from .assemble import labels_of
    cfg, layout = _context(args)
    keys = None
    scope = "cohort"
    if args.pilot:
        keys = _pilot_keys(layout)
        if not keys:
            raise TxBenchError("no pilot on record. Run `bench collect --pilot N` first, or "
                               "drop --pilot to audit the whole cohort.")
        scope = f"pilot({len(keys)})"
    manifest, rel, assembled = _assembled(cfg, layout, args, keys)
    checks = audit(assembled, args.detector or None)
    smoke = None
    repo = Path(args.blockscan_repo) if args.blockscan_repo else default_repo_dir(layout.root)
    chosen = _smoke_subject(assembled)
    if chosen is not None and repo.is_dir():
        smoke = run_smoke(chosen, repo)
    out_dir = layout.adapters_dir / ("pilot" if args.pilot else "cohort")
    written = write_audit(checks, out_dir, labels=labels_of(manifest),
                          extras={"scope": scope, "blockscan_smoke_test": smoke,
                                  "records_assembled": len(assembled),
                                  "records_missing": sum(1 for _, tx in assembled if tx is None),
                                  "recovered_call_trees":
                                      getattr(_assembled, "last_tree_application", None)})
    from .adapters import coverage_table
    _emit({"scope": scope, "coverage": coverage_table(checks, labels_of(manifest)),
           "recovered_call_trees": getattr(_assembled, "last_tree_application", None),
           "blockscan_smoke_test": smoke,
           "artifacts": {k: str(v) for k, v in written.items()}})
    return 0


def _smoke_subject(assembled):
    """The record to run the authors' preprocessor on.

    A flat record makes the smoke test report `degraded`, which says nothing
    about whether real nesting survives the conversion - the one thing worth
    testing. So a record whose recovered tree has at least one child is
    preferred, and the first assembled record is only the fallback. Both
    choices are deterministic: `assembled` is in manifest order.
    """
    fallback = None
    for _, tx in assembled:
        if tx is None:
            continue
        if fallback is None:
            fallback = tx
        if tx.call_tree_has_nesting and any(f.parent_index is not None for f in tx.frames):
            return tx
    return fallback


def cmd_validate_adapters(args: argparse.Namespace) -> int:
    args.pilot = False
    return cmd_audit_inputs(args)


def cmd_build_features(args: argparse.Namespace) -> int:
    from . import features as F
    from .assemble import labels_of
    from .splits import TRAIN, VALIDATION, TEST, load_partitions
    cfg, layout = _context(args)
    view = args.view
    if view not in F.VIEWS:
        raise TxBenchError(f"unknown view {view!r}; choose from {list(F.VIEWS)}")
    manifest, rel, assembled = _assembled(cfg, layout, args)
    labels = labels_of(manifest)
    sparse = F.build_sparse_view(view, assembled)

    split_name = args.split or cfg.features.fit_split
    # `primary_with_validation.parquet` is authoritative: `verify-splits`
    # builds it on top of a corrected assignment when one was needed.
    partitions_path = layout.splits_dir / "primary_with_validation.parquet"
    if partitions_path.is_file():
        partitions = load_partitions(partitions_path)
        partition_source = str(partitions_path)
    else:
        column = f"split_{split_name}"
        if column not in manifest.columns:
            raise TxBenchError(f"no partitions on disk and no {column} column; run "
                               "`bench verify-splits` first.")
        partitions = {row.tx_key: getattr(row, column) for row in manifest.itertuples(index=False)
                      if getattr(row, column)}
        partition_source = f"manifest column {column} (no validation partition; run "\
                           "`bench verify-splits` to create one)"
        log.warning("using upstream %s directly: no validation partition exists yet", column)

    train_keys = [k for k, p in partitions.items() if p == TRAIN]
    vocab = F.fit_vocabulary(sparse, train_keys, split=split_name,
                             min_train_frequency=cfg.features.min_train_frequency)

    out_dir = F.view_dir(layout.features_dir, view)
    written: dict[str, Path] = {}
    written["tokens"] = F.write_parquet(sparse.to_long_frame(), out_dir / "tokens.parquet")
    written["row_status"] = F.write_parquet(sparse.to_status_frame(), out_dir / "row_status.parquet")
    written["vocabulary"] = F.write_json(vocab.as_dict(), out_dir / f"vocab_{split_name}.json")

    matrices: dict[str, Any] = {}
    for partition in (TRAIN, VALIDATION, TEST):
        keys = sorted(k for k, p in partitions.items() if p == partition)
        if not keys:
            continue
        matrix = F.build_matrix(sparse, vocab, keys, labels, partition)
        path = out_dir / f"matrix_{split_name}_{partition}.parquet"
        written[f"matrix_{partition}"] = F.write_parquet(matrix, path)
        matrices[partition] = {"rows": int(len(matrix)),
                               "usable_rows": int(matrix["usable"].sum()),
                               "attacks": int((matrix["label"] == 1).sum()),
                               "negatives": int((matrix["label"] == 0).sum())}
        if partition == TRAIN:
            written["scalers"] = F.write_json(F.fit_scalers(matrix, vocab),
                                              out_dir / f"scalers_{split_name}.json")
    dictionary = F.feature_dictionary(view, vocab)
    dictionary["partition_source"] = partition_source
    dictionary["partitions"] = matrices
    dictionary["coverage"] = F.coverage(sparse, labels)
    written["feature_dictionary"] = F.write_json(dictionary, out_dir / "feature_dictionary.json")
    _emit({"view": view, "split": split_name, "vocabulary_tokens": len(vocab.tokens),
           "coverage": dictionary["coverage"], "partitions": matrices,
           "artifacts": {k: str(v) for k, v in written.items()}})
    return 0


def cmd_verify_splits(args: argparse.Namespace) -> int:
    from .splits import verify, write_splits
    cfg, layout = _context(args)
    manifest = _manifest(layout)
    report = verify(manifest, primary=cfg.splits.primary, secondary=cfg.splits.secondary,
                    metadata_only=cfg.splits.metadata_only,
                    validation_fraction=cfg.splits.validation_fraction)
    written = write_splits(report, layout.splits_dir)
    _emit({"primary": report.get("primary"),
           "validation_partition": report.get("validation_partition"),
           "corrected_split": report.get("corrected_split"),
           "audits": {k: {"clean": v["clean"], "partitions": v["partitions"],
                          "group_overlap_counts": v["group_overlap_counts"],
                          "chronology": v.get("chronology")}
                      for k, v in report["audits"].items()},
           "artifacts": {k: str(v) for k, v in written.items()}})
    return 0


def cmd_build_tasks(args: argparse.Namespace) -> int:
    from .tasks import build_tasks, load_vulnerability_index, write_tasks
    cfg, layout = _context(args)
    manifest = _manifest(layout)
    index = None
    try:
        rel = _release(cfg, layout, args)
        if rel.has("accepted_attacks"):
            index = load_vulnerability_index(rel.path("accepted_attacks"))
    except TxBenchError as exc:
        log.warning("no release available for the task mapping: %s", redact(exc))
    tasks, decisions = build_tasks(manifest, vulnerability_index=index)
    written = write_tasks(tasks, decisions, layout.eval_dir / "tasks")
    _emit({"tasks": {k: v.as_dict() for k, v in tasks.items()},
           "artifacts": {k: str(v) for k, v in written.items()}})
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from .assemble import attack_classes_of, incidents_of, labels_of
    from . import evaluate as E
    from .splits import TEST, VALIDATION, load_partitions
    from .tasks import TASK_EXPLOIT, build_tasks, load_vulnerability_index
    cfg, layout = _context(args)
    manifest = _manifest(layout)
    labels = labels_of(manifest)
    incidents = incidents_of(manifest)
    attack_classes = attack_classes_of(manifest)

    partitions_path = layout.splits_dir / "primary_with_validation.parquet"
    if not partitions_path.is_file():
        raise TxBenchError("no split partitions. Run `bench verify-splits` first.")
    partitions = load_partitions(partitions_path)

    index = None
    rel = None
    try:
        rel = _release(cfg, layout, args)
        if rel.has("accepted_attacks"):
            index = load_vulnerability_index(rel.path("accepted_attacks"))
    except TxBenchError as exc:
        log.warning("no release available: %s", redact(exc))
    tasks, _ = build_tasks(manifest, vulnerability_index=index)
    task = tasks[args.task]

    eligible = task.eligible
    sources = [Path(p) for p in args.predictions]
    if len(sources) > 1:
        from .models import merge_predictions_csv
        predictions_path = merge_predictions_csv(
            sources, layout.eval_dir / "predictions" / "predictions_merged.csv")
    else:
        predictions_path = sources[0]
    bundles = E.read_predictions(predictions_path, partitions=partitions,
                                 eligible=eligible if args.strict_eligible else None)
    results: dict[str, Any] = {
        "task": task.as_dict(),
        "fixture_mode": bool(args.fixture_mode),
        "is_fixture_data": bool(manifest["is_fixture"].any()) if len(manifest) else None,
        "models": {},
    }
    if args.fixture_mode:
        results["warning"] = ("fixture mode: these numbers describe a prediction fixture and "
                              "justify no accuracy claim about any detector")

    validation_keys = [k for k in eligible if partitions.get(k) == VALIDATION]
    test_keys = [k for k in eligible if partitions.get(k) == TEST]
    # A threshold chosen on one task's validation rows can be *carried* to
    # another task rather than chosen again: where a task's validation
    # partition holds none of its positives, choosing on it would be choosing
    # on nothing. The source is recorded with the number.
    frozen: dict[str, Any] = {}
    if args.thresholds_from:
        source = Path(args.thresholds_from)
        if not source.is_file():
            raise TxBenchError(f"no evaluation artifact at {source} to take thresholds from")
        earlier = json.loads(source.read_text(encoding="utf-8"))
        for name, block in (earlier.get("models") or {}).items():
            selection = block.get("threshold_selection") or {}
            frozen[name] = {"threshold": selection.get("threshold"),
                            "rule": selection.get("rule"),
                            "frozen_from": f"{source} (task {earlier.get('task', {}).get('task')})",
                            "fitted_on": selection.get("fitted_on")}
    # A prediction file deliberately covers the whole frozen set, so rows outside
    # *this* task's eligible hashes are not stray: they belong to another task.
    # Saying so is better than letting "unexpected rows" imply a mistake.
    other_eligible = {k for other, definition in tasks.items() if other != args.task
                      for k in definition.eligible}
    thresholds: dict[str, float | None] = {}
    for name, bundle in bundles.items():
        completeness = E.check_completeness(bundle, eligible)
        unexpected = set(bundle.predictions) - set(eligible)
        completeness["unexpected_rows_eligible_in_another_task"] = len(
            unexpected & other_eligible)
        completeness["unexpected_rows_eligible_nowhere"] = len(unexpected - other_eligible)
        completeness["complete_over_this_task"] = not completeness["missing_rows"]
        if name in frozen:
            chosen = dict(frozen[name])
            chosen["carried"] = True
        else:
            chosen = E.select_threshold(bundle, labels, validation_keys,
                                        rule=args.threshold_rule,
                                        target_fp_per_1k=args.target_fp_per_1k)
            chosen["carried"] = False
        thresholds[name] = chosen.get("threshold")
        results["models"][name] = {
            "completeness": completeness,
            "threshold_selection": chosen,
            "validation": E.score_partition(bundle, labels, validation_keys,
                                            threshold=chosen.get("threshold"),
                                            incidents=incidents, partition="validation",
                                            attack_classes=attack_classes),
            "test": E.score_partition(bundle, labels, test_keys,
                                      threshold=chosen.get("threshold"),
                                      incidents=incidents, partition="test",
                                      attack_classes=attack_classes),
        }
    if len(bundles) > 1:
        results["paired_comparison"] = E.paired_comparison(
            results["models"], bundles, labels, test_keys, thresholds=thresholds,
            incidents=incidents, attack_classes=attack_classes)
    results["split_counts"] = {
        "eligible": len(eligible),
        "validation": {"rows": len(validation_keys),
                       "attacks": sum(1 for k in validation_keys if labels.get(k) == 1),
                       "negatives": sum(1 for k in validation_keys if labels.get(k) == 0)},
        "test": {"rows": len(test_keys),
                 "attacks": sum(1 for k in test_keys if labels.get(k) == 1),
                 "negatives": sum(1 for k in test_keys if labels.get(k) == 0)},
        "train_rows_in_split": sum(1 for k in eligible if partitions.get(k) == "train"),
    }

    parity: dict[str, Any] = {"status": "unverified",
                              "reason": "no pinned upstream evaluator was available"}
    if rel is not None and not args.no_parity and rel.has("evaluator") and rel.has("combined"):
        work = layout.eval_dir / "parity"
        E.write_upstream_benchmark_subset(rel.path("combined"), work, chain_id=cfg.chain_id)
        name, bundle = next(iter(bundles.items()))
        upstream_predictions = E.to_upstream_predictions(
            bundle, eligible, thresholds.get(name), work / "upstream_predictions.csv")
        parity = E.upstream_parity(rel.path("evaluator"), work, upstream_predictions,
                                   split=args.split, track=args.track,
                                   base_rate=args.base_rate,
                                   tolerance=args.tolerance,
                                   python_executable=sys.executable)
        parity["model_compared"] = name
        parity["note"] = ("the upstream evaluator counts a row with no prediction as not "
                          "flagged; this project's own metrics instead require an explicit "
                          "row per eligible hash")
    results["upstream_evaluator_parity"] = parity

    destination = (Path(args.out) if args.out else layout.eval_dir / "evaluation.json")
    if not destination.is_absolute():
        destination = layout.root / destination
    out = E.write_json(results, destination)
    _emit({"task": args.task, "models": sorted(bundles),
           "parity_status": parity.get("status"),
           "test": {k: {"recall": v["test"]["recall"], "precision": v["test"]["precision"],
                        "average_precision": v["test"]["average_precision"],
                        "fp_per_1000_negatives": v["test"]["false_positives_per_1000_negatives"]}
                    for k, v in results["models"].items()},
           "artifact": str(out)})
    return 0


def cmd_enrich_traces(args: argparse.Namespace) -> int:
    """Recover call trees for the transactions a missing call path is blocking.

    Order matters and is enforced: pick the pilot from the coverage output,
    inspect the table's real schema, price the scan, and only then - with a
    byte budget configured - execute anything.
    """
    from . import bq
    from .adapters.blockscan_smoke import default_repo_dir
    from .cache import CacheStore
    from .enrich import (cached_rows, jsonable_rows, partition_cached, rebuild_one,
                         rows_to_traces, select_pilot, store_rows, tracer_config_for,
                         write_enrichment_report)
    cfg, layout = _context(args)
    manifest = _manifest(layout)
    table = args.table or cfg.bigquery.table
    out_dir = layout.data / "collection"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. the pilot --------------------------------------------------------
    pilot = select_pilot(layout, manifest, detector=args.detector,
                         expected=args.expect, scope=args.scope, select=args.select)
    discrepancy = pilot.discrepancy()
    if discrepancy and not args.allow_count_mismatch:
        _emit({"step": "select_pilot", "pilot": pilot.summary(), "discrepancy": discrepancy,
               "action": "nothing was queried. Re-run `bench validate-adapters`, or pass "
                         "--allow-count-mismatch to proceed with the listed hashes."})
        return 2

    store = CacheStore(layout.cache_dir, layout.cache_index)
    hits, misses = partition_cached(pilot, store, cfg.chain_id, table)
    pilot_block = {"pilot": pilot.summary(), "discrepancy": discrepancy,
                   "cached_hashes": len(hits), "hashes_to_fetch": len(misses)}
    # Step 1 is complete and offline, so it is written out before anything
    # that can fail: a credential problem must not cost the user the pilot.
    # A different selection writes a different file. Otherwise pricing the
    # full set would overwrite a completed pilot's selection record - and a
    # dry run must never disturb what an earlier run produced.
    pilot_path = out_dir / ("trace_backfill_pilot.parquet" if args.select == "blocked"
                            else f"trace_backfill_selection_{args.select}.parquet")
    tmp = pilot_path.with_suffix(".parquet.tmp")
    pilot.frame.to_parquet(tmp, index=False)
    tmp.replace(pilot_path)
    pilot_block["artifacts"] = {"pilot_table": str(pilot_path)}

    # --- 2. the source, its schema and its price -----------------------------
    schema_block: dict[str, Any] = {"inspected": False}
    query = None
    estimate: int | None = 0
    client = None
    if misses and not args.rebuild_only:
        try:
            client = bq.load_client(args.project or _project_from_env(cfg))
        except bq.BigQueryUnavailable as exc:
            _emit({"step": "inspect_source", **pilot_block,
                   "result": "step 1 is complete; step 2 needs BigQuery access.",
                   "blocked_on": str(exc),
                   "unblock": [
                       "conda run -n txbench python -m pip install -e '.[bigquery]'",
                       "gcloud auth application-default login",
                       f"set {cfg.bigquery.project_var} in .env to the billing project",
                   ],
                   "then": "conda run -n txbench bench enrich-traces --expect 42 --dry-run"})
            return 4
        description = bq.describe_table(client, table)
        resolved = bq.resolve_columns(description)
        schema_block = {"inspected": True, "table": description.as_dict(),
                        "column_mapping": resolved}
        (out_dir / "bigquery_schema.json").write_text(
            json.dumps(schema_block, indent=2, sort_keys=True, default=str))
        if not resolved["usable_for_call_tree"]:
            _emit({"step": "inspect_source", **pilot_block, "schema": schema_block,
                   "result": "the table has no usable call path, so no call tree can be built "
                             "from it. Nothing was scanned.",
                   "artifacts": {**pilot_block["artifacts"],
                                 "schema": str(out_dir / "bigquery_schema.json")}})
            return 3
        blocks_by_date = pilot.blocks_by_date(misses)
        query = bq.build_query(description, resolved, hashes=misses,
                               dates=sorted(blocks_by_date), blocks_by_date=blocks_by_date)
        priced = bq.dry_run(client, query)
        estimate = priced["estimated_bytes_processed"]
        schema_block["dry_run"] = priced
        schema_block["query"] = query.as_dict()
        (out_dir / "bigquery_schema.json").write_text(
            json.dumps(schema_block, indent=2, sort_keys=True, default=str))

    # --- 3. the budget gate --------------------------------------------------
    budget = bq.Budget.load(out_dir / "bigquery_budget.json",
                            args.budget_bytes if args.budget_bytes is not None
                            else cfg.bigquery.total_byte_budget)
    executed: dict[str, Any] | None = None
    if misses and not args.rebuild_only:
        if args.dry_run or not budget.configured:
            suggested = (max(int(estimate * 2), 1 << 30) if estimate is not None
                         else 16 << 30)
            # The command has to reproduce *this* run, so every argument that
            # changed what was priced is echoed back into it.
            parts = ["conda run -n txbench bench enrich-traces"]
            if args.select != "blocked":
                parts.append(f"--select {args.select}")
            if args.scope != "cohort":
                parts.append(f"--scope {args.scope}")
            if args.detector != "defiguard":
                parts.append(f"--detector {args.detector}")
            if args.expect:
                parts.append(f"--expect {args.expect}")
            parts.append(f"--budget-bytes {suggested}")
            command = " ".join(parts)
            _emit({"step": "dry_run", **pilot_block, "schema": schema_block,
                   "budget": budget.snapshot(),
                   "result": ("priced only; nothing was scanned."
                              if args.dry_run else
                              "no byte budget is configured, so nothing was scanned."),
                   "next_command": command,
                   "artifacts": {**pilot_block["artifacts"],
                                 "schema": str(out_dir / "bigquery_schema.json")}})
            return 0
        executed = bq.execute(client, query, budget=budget, estimate=estimate,
                              label=f"{args.detector}-backfill")
        budget.save()
        # The bytes are already billed at this point, so nothing after it may
        # throw: BigQuery's datetime, Decimal and bytes objects are normalised
        # here, before anything tries to cache or read them.
        executed["rows"] = jsonable_rows(executed["rows"])
        grouped = rows_to_traces(executed["rows"])
        for tx_hash in misses:
            store_rows(store, cfg.chain_id, tx_hash, table,
                       [r for r in executed["rows"]
                        if str(r.get("tx_hash", "")).lower() == tx_hash],
                       block_number=None)
        log.info("cached warehouse rows for %d hash(es); %d matched",
                 len(misses), len(grouped))

    # --- 4. rebuild ----------------------------------------------------------
    keys = pilot.frame["tx_key"].tolist()
    _, _, assembled = _assembled(cfg, layout, args, keys)
    by_key = dict(assembled)
    repo = Path(args.blockscan_repo) if args.blockscan_repo else default_repo_dir(layout.root)
    rebuilds = []
    for row in pilot.frame.itertuples(index=False):
        rows = cached_rows(store, cfg.chain_id, row.tx_hash, table)
        rebuilds.append(rebuild_one(by_key.get(row.tx_key), rows, tx_key=row.tx_key,
                                    tx_hash=row.tx_hash, table=table,
                                    blockscan_repo=repo if repo.is_dir() else None))

    # --- 5. report -----------------------------------------------------------
    written = write_enrichment_report(
        pilot, rebuilds, out_dir,
        extras={
            "source_table": table,
            "cache_tracer_config": tracer_config_for(table),
            "schema": schema_block,
            "estimated_bytes": estimate,
            "actual_bytes_billed": (executed or {}).get("bytes_billed", 0),
            "maximum_bytes_billed": (executed or {}).get("maximum_bytes_billed"),
            "bigquery_job_id": (executed or {}).get("job_id"),
            "cached_hashes_reused": len(hits),
            "hashes_fetched_this_run": 0 if executed is None else len(misses),
            "budget": budget.snapshot(),
            "rebuild_only": bool(args.rebuild_only),
            "blockscan_repo_present": repo.is_dir(),
        })
    report = json.loads(written["report"].read_text())
    _emit({"step": "complete", **{k: report[k] for k in (
        "requested_hashes", "hashes_matched_in_bigquery", "trees_recovered_complete",
        "defiguard_graphs_unblocked", "defiguard_graphs_unblocked_of_requested",
        "reverted_ancestry_resolved", "blockscan_preprocess_ok", "estimated_bytes",
        "actual_bytes_billed", "cached_hashes_reused", "missing_reasons")},
        "artifacts": {k: str(v) for k, v in written.items()}})
    return 0


def cmd_backfill_init(args: argparse.Namespace) -> int:
    """Fetch creation init bytes for the frames that still lack them.

    Targeted on purpose: three columns, only the creation frames, only the
    transactions whose assembled record is still short. `result.code` is not
    fetched - varying it through the BlockScan adapter produced a
    byte-identical token stream, so it would buy nothing readable here.
    """
    import pandas as pd

    from . import bq
    from .cache import CacheStore
    from .enrich import (hashes_missing_creation_input, jsonable_rows, patch_cached_init,
                         select_init_gaps, tracer_config_for)
    cfg, layout = _context(args)
    table = args.table or cfg.bigquery.table
    out_dir = layout.data / "collection"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, _, assembled = _assembled(cfg, layout, args)
    wanted = hashes_missing_creation_input(assembled)
    store = CacheStore(layout.cache_dir, layout.cache_index)
    gaps = select_init_gaps(store, cfg.chain_id, table, wanted)
    frames = sum(len(paths) for paths in gaps.values())
    plan = {"hashes_missing_input": len(wanted), "hashes_with_cached_frames": len(gaps),
            "frames_to_patch": frames}
    if not frames:
        _emit({"step": "complete", **plan,
               "result": "no creation frame is missing its init bytes; nothing to fetch."})
        return 0

    manifest = _manifest(layout)
    rows = manifest[manifest["tx_hash"].str.lower().isin(gaps)].copy()
    stamps = pd.to_datetime(rows["block_timestamp"], unit="s", errors="coerce", utc=True)
    rows["block_date"] = stamps.dt.strftime("%Y-%m-%d")
    if rows["block_date"].isna().any():
        raise TxBenchError("some affected transactions have no usable timestamp")
    blocks_by_date: dict[str, list[int]] = {}
    for row in rows.itertuples(index=False):
        blocks_by_date.setdefault(row.block_date, []).append(int(row.block_number))
    blocks_by_date = {d: sorted(set(b)) for d, b in sorted(blocks_by_date.items())}

    try:
        client = bq.load_client(args.project or _project_from_env(cfg))
    except bq.BigQueryUnavailable as exc:
        _emit({"step": "inspect_source", **plan, "blocked_on": str(exc)})
        return 4
    description = bq.describe_table(client, table)
    resolved = bq.resolve_columns(description)
    query = bq.build_init_query(description, resolved, hashes=sorted(gaps),
                                dates=sorted(blocks_by_date), blocks_by_date=blocks_by_date)
    priced = bq.dry_run(client, query)
    estimate = priced["estimated_bytes_processed"]
    plan |= {"dry_run": priced, "distinct_days": len(blocks_by_date),
             "distinct_blocks": sum(len(b) for b in blocks_by_date.values())}

    budget = bq.Budget.load(out_dir / "bigquery_budget.json",
                            args.budget_bytes if args.budget_bytes is not None
                            else cfg.bigquery.total_byte_budget)
    if args.dry_run or not budget.configured:
        _emit({"step": "dry_run", **plan, "budget": budget.snapshot(),
               "result": ("priced only; nothing was scanned." if args.dry_run else
                          "no byte budget is configured, so nothing was scanned."),
               # --budget-bytes is the ledger's *lifetime* cap, not a per-run
               # allowance, so the suggestion has to clear what is already spent
               # or the job is refused for being under water.
               "next_command": ("conda run -n txbench bench backfill-init --budget-bytes "
                                f"{budget.spent_bytes + max(int((estimate or 0) * 2), 1 << 30)}"
                                if budget.configured else
                                "conda run -n txbench bench backfill-init --budget-bytes "
                                f"{max(int((estimate or 0) * 2), 1 << 30)}"),
               "budget_note": ("--budget-bytes sets the total this ledger may ever bill; "
                               f"{budget.spent_bytes} byte(s) are already spent, so a "
                               "smaller figure is refused rather than applied per run.")})
        return 0

    executed = bq.execute(client, query, budget=budget, estimate=estimate,
                          label="creation-init-backfill")
    budget.save()
    fetched = jsonable_rows(executed["rows"])
    patched = patch_cached_init(store, cfg.chain_id, table, fetched)
    report = {"step": "complete", **plan,
              "rows_returned": len(fetched),
              "estimated_bytes": estimate,
              "actual_bytes_billed": executed["bytes_billed"],
              "maximum_bytes_billed": executed["maximum_bytes_billed"],
              "bigquery_job_id": executed["job_id"],
              "cache_tracer_config": tracer_config_for(table),
              "patch": patched,
              "budget": budget.snapshot()}
    path = out_dir / "init_backfill_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    _emit({**report, "artifacts": {"report": str(path)},
           "then": "conda run -n txbench bench validate-adapters"})
    return 0


def _project_from_env(cfg) -> str | None:
    """The billing project, from the variable the config names. Never a key."""
    import os
    return os.environ.get(cfg.bigquery.project_var) or None


def cmd_score_blockscan(args: argparse.Namespace) -> int:
    """Score the frozen set with BlockScan's own tokenizer, checkpoint and scorer.

    Nothing is approximated: if one of the authors' assets is absent the run
    stops and names it, with where it comes from. The smoke test runs first and
    must prove two things on one training transaction - a finite score, and a
    recovered nested call present in the ids the model was actually fed.
    """
    import math

    import pandas as pd

    from .adapters.blockscan_smoke import default_repo_dir, load_entry_module
    from .models import blockscan_run as B
    from .models import PredictionRow, write_json, write_predictions_csv
    from .splits import load_partitions

    cfg, layout = _context(args)
    repo_dir = Path(args.blockscan_repo) if args.blockscan_repo else default_repo_dir(layout.root)
    assets = B.asset_report(repo_dir, checksums=True)
    if not assets["complete"]:
        detail = "; ".join(f"{m['asset']} missing at {m['path']} (needed by {m['needed_by']}; "
                           f"source: {m['source']})" for m in assets["missing"])
        raise TxBenchError(f"BlockScan cannot run: {detail}")
    scorer_python = B.resolve_scorer_python(args.scorer_python)

    partitions_path = layout.splits_dir / "primary_with_validation.parquet"
    if not partitions_path.is_file():
        raise TxBenchError("no split partitions. Run `bench verify-splits` first.")
    partitions = load_partitions(partitions_path)
    manifest, _rel, assembled = _assembled(cfg, layout, args)
    preprocess = load_entry_module(repo_dir)
    allow_list = B.load_allow_list(repo_dir)

    work = Path(args.work_dir) if args.work_dir else layout.data / "detectors" / "blockscan"
    # A sweep writes its runs elsewhere so the primary seed-42 artifacts in
    # data/evaluation/detectors stay exactly as they were.
    out_dir = (Path(args.out_dir) if getattr(args, "out_dir", None)
               else layout.eval_dir / "detectors")
    out_dir.mkdir(parents=True, exist_ok=True)
    run: dict[str, Any] = {
        "model": B.MODEL_NAME, "version": B.MODEL_VERSION,
        "variant": "the authors' pretrained foundation model, run zero-shot: it is not "
                   "fine-tuned on this project's transactions, and no label is read at "
                   "scoring time",
        "assets": assets,
        "scorer_python": str(scorer_python),
        "entry_points": ["src_eth/preprocess.py::write_all_traces",
                         "src_eth/detection.py::compute_abnormal_score"],
        "oov_allow_list_entries": len(allow_list),
        "recovered_call_trees": getattr(_assembled, "last_tree_application", None),
        "seed": args.seed,
        "scorer_settings": {"mask_percent": args.mask_percent, "top_s": args.top_s,
                            "seq_len": args.seq_len, "random_masking": True},
    }

    # --- the smoke test, first and non-optional -------------------------------
    subject = B.choose_smoke_subject(assembled, partitions, allow_list,
                                     partition=args.smoke_partition)
    smoke_in = work / "smoke_input"
    smoke_out = work / "smoke_preprocessed"
    B.clear_directory(smoke_in)
    B.clear_directory(smoke_out)
    subject_record = [(k, tx) for k, tx in assembled if k == subject["tx_key"]]
    staged, failures = B.stage_records(subject_record, partitions, smoke_in,
                                       extract_function_calls=preprocess.extract_function_calls)
    if failures or not staged:
        raise TxBenchError(f"could not stage the smoke subject {subject['tx_key']}")
    B.run_pinned_preprocess(preprocess, input_dir=smoke_in, config_dir=repo_dir / "config",
                            output_dir=smoke_out)
    slot = next(iter(staged.values()))
    smoke_scores = smoke_out / slot.partition / "scores.json"
    decoded_path = smoke_out / slot.partition / "model_input_decoded.json"
    payload = B.score_partition(scorer_python, repo_dir=repo_dir,
                                data_dir=smoke_out / slot.partition, out_json=smoke_scores,
                                seq_len=args.seq_len, mask_percent=args.mask_percent,
                                top_s=args.top_s, seed=args.seed, decode_out=decoded_path)
    rows = B.rows_from_scores(payload, chain_id=cfg.chain_id, partition=slot.partition,
                              raw_token_counts=slot.raw_token_counts)
    decoded = json.loads(decoded_path.read_text(encoding="utf-8"))
    subject_tx = dict(subject_record)[subject["tx_key"]]
    evidence = B.nested_call_evidence(subject, next(iter(decoded.values())),
                                      logs=len(subject_tx.logs))
    score = rows[0].score if rows else None
    finite = score is not None and math.isfinite(float(score))
    smoke = {"score": score, "score_is_finite": finite, "status": rows[0].status if rows else None,
             "tokens": {k: rows[0].extra.get(k) for k in
                        ("tokens_before_truncation", "tokens_scored", "masked_tokens",
                         "truncated", "trace_tokens_before_oov_removal",
                         "trace_tokens_after_oov_removal", "oov_tokens_removed")},
             "nested_call_evidence": evidence,
             "model": payload["model"], "tokenizer": payload["tokenizer"],
             "environment": payload["environment"]}
    run["smoke_test"] = smoke
    if not finite:
        run["smoke_test"]["verdict"] = "failed: the pinned scorer produced no finite score"
        write_json(run, out_dir / "blockscan_run.json")
        raise TxBenchError("BlockScan smoke test failed: no finite score; nothing was scored")
    if not evidence["nested_call_reached_model"]:
        run["smoke_test"]["verdict"] = ("failed: no nested call survived OOV removal and "
                                        "truncation, so the model never saw one")
        write_json(run, out_dir / "blockscan_run.json")
        raise TxBenchError("BlockScan smoke test failed: no nested call reached the model input")
    run["smoke_test"]["verdict"] = "passed"
    _emit({"smoke_test": smoke})
    if args.smoke_only:
        run["partitions_scored"] = []
        artifact = write_json(run, out_dir / "blockscan_smoke.json")
        _emit({"artifact": str(artifact)})
        return 0

    # --- the full run ---------------------------------------------------------
    wanted = [p.strip() for p in args.partitions.split(",") if p.strip()]
    selected = [(k, tx) for k, tx in assembled if partitions.get(k) in wanted]
    if args.limit:
        selected = selected[:args.limit]
    stage_dir = work / "input"
    pre_dir = work / "preprocessed"
    B.clear_directory(stage_dir)
    B.clear_directory(pre_dir)
    staged, stage_failures = B.stage_records(selected, partitions, stage_dir,
                                             extract_function_calls=preprocess.extract_function_calls)
    B.run_pinned_preprocess(preprocess, input_dir=stage_dir, config_dir=repo_dir / "config",
                            output_dir=pre_dir)
    all_rows = []
    per_partition = {}
    for partition, slot in sorted(staged.items()):
        payload = B.score_partition(scorer_python, repo_dir=repo_dir,
                                    data_dir=pre_dir / partition,
                                    out_json=pre_dir / partition / "scores.json",
                                    seq_len=args.seq_len, mask_percent=args.mask_percent,
                                    top_s=args.top_s, seed=args.seed)
        rows = B.rows_from_scores(payload, chain_id=cfg.chain_id, partition=partition,
                                  raw_token_counts=slot.raw_token_counts)
        all_rows.extend(rows)
        # The seed as the scoring process itself held it. detection.py calls
        # np.random.seed(42) at import, so a requested seed only governs the
        # masking because the scorer reseeds afterwards; this records the value
        # that was actually in force rather than the one asked for.
        settings = payload.get("scorer_settings") or {}
        per_partition[partition] = {
            "effective_numpy_seed": settings.get("numpy_seed_effective"),
            "numpy_seed_before_detection_import":
                settings.get("numpy_seed_before_detection_import"),
            "transactions": len(rows),
            "scored": sum(1 for r in rows if r.status == "scored"),
            "failed": sum(1 for r in rows if r.status != "scored"),
            "truncation": B.truncation_summary(rows),
            "score_range": [min((r.score for r in rows if r.score is not None), default=None),
                            max((r.score for r in rows if r.score is not None), default=None)],
        }
    for tx_key in stage_failures:
        all_rows.append(PredictionRow(
            tx_key=tx_key, model=B.MODEL_NAME, version=B.MODEL_VERSION,
            split=partitions.get(tx_key) or "", status="skipped",
            note="no normalised record, or no partition in the frozen split"))
    predictions = write_predictions_csv(all_rows, out_dir / "predictions_blockscan.csv")
    frame = pd.DataFrame([r.as_record() for r in all_rows])
    frame.to_parquet(out_dir / "predictions_blockscan.parquet", index=False)
    run["partitions_scored"] = per_partition
    run["rows"] = len(all_rows)
    run["artifacts"] = {"predictions_csv": str(predictions),
                        "predictions_parquet": str(out_dir / "predictions_blockscan.parquet")}
    artifact = write_json(run, out_dir / "blockscan_run.json")
    _emit({"model": f"{B.MODEL_NAME}@{B.MODEL_VERSION}", "rows": len(all_rows),
           "partitions": {k: v["scored"] for k, v in per_partition.items()},
           "truncated": {k: v["truncation"]["truncated_transactions"]
                         for k, v in per_partition.items()},
           "artifacts": run["artifacts"], "report": str(artifact)})
    return 0


def cmd_train_defiguard(args: argparse.Namespace) -> int:
    """Train the paper-based DeFiGuard reimplementation on the frozen split.

    The three available node features go in; `node_type` is left out of the
    input dimension rather than defaulted, because a zero there would be a
    claim about every address in the set. A transaction whose flow graph is
    incomplete is reported as unscorable, with the reason, and is not trained
    on.
    """
    import math

    import pandas as pd
    import torch

    from .assemble import labels_of
    from .models import defiguard_gnn as G
    from .models import PredictionRow, write_json, write_predictions_csv
    from .splits import load_partitions

    cfg, layout = _context(args)
    manifest = _manifest(layout)
    labels = labels_of(manifest)
    partitions_path = layout.splits_dir / "primary_with_validation.parquet"
    if not partitions_path.is_file():
        raise TxBenchError("no split partitions. Run `bench verify-splits` first.")
    partitions = load_partitions(partitions_path)
    _m, _rel, assembled = _assembled(cfg, layout, args)

    graphs, excluded = G.build_graphs(assembled, partitions)
    by_partition: dict[str, list[Any]] = {}
    for graph in graphs:
        by_partition.setdefault(partitions.get(graph.tx_key) or "", []).append(graph)
    counts = {k: len(v) for k, v in sorted(by_partition.items())}
    excluded_rows = [{"tx_key": e.tx_key, "partition": e.partition, "outcome": e.outcome,
                      "reason": e.reason} for e in excluded]
    for row in excluded_rows:
        log.warning("defiguard: excluding %s (%s): %s", row["tx_key"], row["partition"],
                    row["reason"])

    # --- a forward pass on a real graph, before any training ------------------
    probe_graph = next((g for g in by_partition.get("train", []) if g.n_edges), None)
    if probe_graph is None:
        raise TxBenchError("no training graph carries an edge; there is nothing to train on")
    probe_model = G.DeFiGuardGNN(hidden=args.hidden, layers=args.layers,
                                 dropout=args.dropout, seed=args.seed).eval()
    with torch.no_grad():
        probe_logit = float(probe_model.forward(G.collate([probe_graph]))[0])
    forward_check = {
        "tx_key": probe_graph.tx_key, "nodes": probe_graph.n_nodes,
        "edges": probe_graph.n_edges, "outcome": probe_graph.outcome,
        "node_feature_dim": int(probe_graph.x.shape[1]),
        "logit": probe_logit, "finite": math.isfinite(probe_logit),
        "ran_before_training": True,
    }
    if not forward_check["finite"]:
        raise TxBenchError("the DeFiGuard forward pass on a real graph produced no finite value")
    _emit({"forward_pass_check": forward_check})

    fitted = G.train(by_partition, labels, hidden=args.hidden, layers=args.layers,
                     dropout=args.dropout, lr=args.lr, weight_decay=args.weight_decay,
                     epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                     seed=args.seed)
    model, scaler = fitted["model"], fitted["scaler"]

    rows: list[PredictionRow] = []
    per_partition: dict[str, Any] = {}
    for partition, slot in sorted(by_partition.items()):
        scaled = G.standardize_apply(slot, scaler)
        scores, keys = G.predict(model, scaled)
        for key, score in zip(keys, scores):
            rows.append(PredictionRow(
                tx_key=key, model=G.MODEL_NAME, version=G.MODEL_VERSION, split=partition,
                status="scored", score=float(score), score_direction="higher_is_attack",
                extra={"graph_nodes": next(g.n_nodes for g in slot if g.tx_key == key),
                       "graph_edges": next(g.n_edges for g in slot if g.tx_key == key),
                       "graph_outcome": next(g.outcome for g in slot if g.tx_key == key)}))
        per_partition[partition] = {"graphs": len(slot), "scored": len(scores)}
    for item in excluded:
        rows.append(PredictionRow(
            tx_key=item.tx_key, model=G.MODEL_NAME, version=G.MODEL_VERSION,
            split=item.partition, status="unsupported_input",
            note=f"flow graph incomplete: {item.reason}",
            extra={"graph_outcome": item.outcome}))

    # A sweep writes its runs elsewhere so the primary seed-42 artifacts in
    # data/evaluation/detectors stay exactly as they were.
    out_dir = (Path(args.out_dir) if getattr(args, "out_dir", None)
               else layout.eval_dir / "detectors")
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = write_predictions_csv(rows, out_dir / "predictions_defiguard.csv")
    pd.DataFrame([r.as_record() for r in rows]).to_parquet(
        out_dir / "predictions_defiguard.parquet", index=False)
    torch.save({"state_dict": model.state_dict(), "config": fitted["config"],
                "scaler": scaler}, out_dir / "defiguard_gnn.pt")
    report = {
        "model": G.MODEL_NAME, "version": G.MODEL_VERSION,
        "variant": "paper-based reimplementation (not the authors' code), three node features, "
                   "node_type omitted from the input dimension",
        "evidence_level": "paper_reimplementation",
        "node_features": list(G.NODE_FEATURES),
        "omitted_node_features": list(G.OMITTED_NODE_FEATURES),
        "graph_features": list(G.GRAPH_FEATURES),
        "forward_pass_check": forward_check,
        "training": {k: v for k, v in fitted.items() if k not in ("model", "history")},
        "history": fitted["history"],
        "graphs_by_partition": counts,
        "excluded": excluded_rows,
        "excluded_count": len(excluded_rows),
        "recovered_call_trees": getattr(_assembled, "last_tree_application", None),
        "scored_by_partition": per_partition,
        "artifacts": {"predictions_csv": str(predictions),
                      "predictions_parquet": str(out_dir / "predictions_defiguard.parquet"),
                      "checkpoint": str(out_dir / "defiguard_gnn.pt")},
    }
    artifact = write_json(report, out_dir / "defiguard_run.json")
    _emit({"model": f"{G.MODEL_NAME}@{G.MODEL_VERSION}",
           "graphs": counts, "excluded": excluded_rows,
           "selected_epoch": fitted["selected_epoch"], "epochs_run": fitted["epochs_run"],
           "validation_average_precision": fitted["validation_ap"],
           "rows": len(rows), "report": str(artifact)})
    return 0


def cmd_train_ft_transformer(args: argparse.Namespace) -> int:
    """Train the FT-Transformer on this project's feature views.

    Both documented views are trained and reported; the one with the higher
    **validation** average precision is the submitted model, chosen before any
    test row is scored.
    """
    import pandas as pd
    import torch

    from .models import ft_transformer as F
    from .models import PredictionRow, write_json, write_predictions_csv
    from .splits import load_partitions

    cfg, layout = _context(args)
    partitions_path = layout.splits_dir / "primary_with_validation.parquet"
    if not partitions_path.is_file():
        raise TxBenchError("no split partitions. Run `bench verify-splits` first.")
    partitions = load_partitions(partitions_path)
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    # A sweep writes its runs elsewhere so the primary seed-42 artifacts in
    # data/evaluation/detectors stay exactly as they were.
    out_dir = (Path(args.out_dir) if getattr(args, "out_dir", None)
               else layout.eval_dir / "detectors")
    out_dir.mkdir(parents=True, exist_ok=True)

    runs: dict[str, Any] = {}
    rows_by_view: dict[str, list[PredictionRow]] = {}
    for view in views:
        data = F.load_view(layout.features_dir, view, split=args.feature_split,
                           min_document_frequency=args.min_document_frequency)
        fitted = F.train(data, d_token=args.d_token, n_blocks=args.blocks,
                         n_heads=args.heads, lr=args.lr, weight_decay=args.weight_decay,
                         batch_size=args.batch_size, epochs=args.epochs,
                         patience=args.patience, seed=args.seed)
        version = f"{F.MODEL_VERSION}+{view}"
        rows: list[PredictionRow] = []
        for partition, (x, _y, keys) in sorted(data.partitions.items()):
            scores = F.predict(fitted["model"], x)
            for key, score in zip(keys, scores):
                declared = partitions.get(key) or partition
                rows.append(PredictionRow(tx_key=key, model=F.MODEL_NAME, version=version,
                                          split=declared, status="scored", score=float(score),
                                          score_direction="higher_is_attack",
                                          extra={"view": view}))
        for partition, keys in data.selection["rows_not_usable_keys"].items():
            for key in keys:
                rows.append(PredictionRow(
                    tx_key=key, model=F.MODEL_NAME, version=version,
                    split=partitions.get(key) or partition, status="unsupported_input",
                    note=f"the {view} view is not usable for this transaction",
                    extra={"view": view}))
        rows_by_view[view] = rows
        torch.save({"state_dict": fitted["model"].state_dict(), "config": fitted["config"],
                    "scaler": data.scaler, "columns": data.columns},
                   out_dir / f"ft_transformer_{view.replace('-', '_')}.pt")
        runs[view] = {
            "selection": data.selection, "scaler": data.scaler,
            "training": {k: v for k, v in fitted.items() if k not in ("model", "history")},
            "history": fitted["history"], "version": version,
            "rows": len(rows),
        }

    ranked = sorted(views, key=lambda v: (runs[v]["training"]["validation_ap"] or -1),
                    reverse=True)
    primary = ranked[0]
    for view in views:
        suffix = view.replace("-", "_")
        path = write_predictions_csv(rows_by_view[view],
                                     out_dir / f"predictions_ft_transformer_{suffix}.csv")
        pd.DataFrame([r.as_record() for r in rows_by_view[view]]).to_parquet(
            out_dir / f"predictions_ft_transformer_{suffix}.parquet", index=False)
        runs[view]["artifacts"] = {"predictions_csv": str(path)}
    primary_csv = write_predictions_csv(
        rows_by_view[primary], out_dir / "predictions_ft_transformer.csv")

    report = {
        "model": F.MODEL_NAME,
        "evidence_level": "project_implementation",
        "variant": "FT-Transformer (arXiv:2106.11959) over this project's own feature views",
        "views": runs,
        "primary_view": primary,
        "primary_version": runs[primary]["version"],
        "primary_selected_on": "validation average precision, before any test row was scored",
        "validation_average_precision_by_view": {
            v: runs[v]["training"]["validation_ap"] for v in views},
        "artifacts": {"predictions_csv": str(primary_csv)},
    }
    artifact = write_json(report, out_dir / "ft_transformer_run.json")
    _emit({"model": F.MODEL_NAME, "primary_view": primary,
           "validation_average_precision": report["validation_average_precision_by_view"],
           "selected_epoch": {v: runs[v]["training"]["selected_epoch"] for v in views},
           "report": str(artifact)})
    return 0


def cmd_analyze_misses(args: argparse.Namespace) -> int:
    """Explain a per-class miss: how few incidents, and against what control.

    A recall of zero over a class says nothing about *why*. Two questions
    separate the readings. How many distinct incidents produced those
    transactions - a class drawn from one event is one miss repeated, not many
    independent failures. And how do the scores compare against negatives of
    the *same transaction shape* - if a class is plain token transfers and the
    negative pool is mostly multi-call DeFi traffic, a detector can look
    informative while only separating simple from complex.

    Read-only. Thresholds come from the evaluation artifact and are applied
    exactly as frozen: this command never selects an operating point.
    """
    import pandas as pd

    from .assemble import attack_classes_of, incidents_of, labels_of
    from . import concentration as C
    from . import evaluate as E
    from .features import view_dir
    from .splits import load_partitions

    cfg, layout = _context(args)
    manifest = _manifest(layout)
    labels = labels_of(manifest)
    incidents = incidents_of(manifest)
    classes = attack_classes_of(manifest)
    partitions = load_partitions(layout.splits_dir / "primary_with_validation.parquet")

    conc = C.concentration_of(manifest, args.attack_class, partition=args.partition)
    log.info("%s: %d transactions from %d incident(s); largest %.1f%%",
             args.attack_class, conc.transactions, conc.incidents,
             100 * conc.largest_incident_share)

    matrix_path = (view_dir(layout.features_dir, args.view)
                   / f"matrix_temporal_{args.partition}.parquet")
    if not matrix_path.is_file():
        raise TxBenchError(f"no feature matrix at {matrix_path}. Run `bench build-features` first.")
    shapes_frame = C.shapes_from_matrix(pd.read_parquet(matrix_path))
    # `.shape` on a DataFrame is its dimensions, not this column.
    shapes = dict(zip(shapes_frame["tx_key"], shapes_frame["shape"]))
    categories = {row.tx_key: str(row.benign_category) for row in
                  manifest.itertuples(index=False)
                  if isinstance(row.benign_category, str) and row.benign_category}

    evaluation = Path(args.thresholds_from)
    if not evaluation.is_file():
        raise TxBenchError(f"no evaluation artifact at {evaluation} to take thresholds from")
    earlier = json.loads(evaluation.read_text(encoding="utf-8"))
    frozen = {name: ((block.get("threshold_selection") or {}).get("threshold"))
              for name, block in (earlier.get("models") or {}).items()}
    if not frozen:
        raise TxBenchError(f"{evaluation} records no thresholds")

    bundles = E.read_predictions(Path(args.predictions), partitions=partitions)
    keys = [k for k, part in partitions.items() if part == args.partition]
    out: dict[str, Any] = {
        "attack_class": args.attack_class,
        "partition": args.partition,
        "feature_view": args.view,
        "thresholds_from": str(evaluation),
        "thresholds_from_task": (earlier.get("task") or {}).get("task"),
        "predictions": str(args.predictions),
        "concentration": conc.to_dict(),
        "shape_definition": {
            "plain_erc20_transfer": (f"{C.SEL_ERC20_TRANSFER} fires and at most "
                                     f"{C.MAX_PLAIN_TOKENS} token features are non-zero"),
            "bare_value_transfer": "no selector and no log topic fires",
            "other": "everything else",
            "source": str(matrix_path),
        },
        "models": {},
    }
    for name, bundle in bundles.items():
        threshold = frozen.get(name)
        if threshold is None:
            log.warning("no frozen threshold for %s: reporting its distribution only", name)
        scores = {k: p.oriented_score for k, p in bundle.predictions.items()
                  if p.oriented_score is not None}
        out["models"][name] = {
            "threshold": threshold,
            "incident_recall": C.incident_recall(
                scores, classes, incidents, attack_class=args.attack_class,
                threshold=threshold),
            "contrast": C.contrast(scores, labels, classes, shapes,
                                   attack_class=args.attack_class,
                                   threshold=threshold, keys=keys,
                                   categories=categories),
        }
    destination = (Path(args.out) if args.out
                   else layout.eval_dir / f"miss_analysis_{args.attack_class}.json")
    C.write_json(out, destination)
    log.info("wrote %s", destination)
    return 0


#: The method labels used in the printed table and the saved artifact.
TOPK_LENGTH_METHOD = "length_baseline (masked_tokens)"

#: AP values established by the earlier score-composition analysis, to three
#: decimals. They are *checks*, not inputs: if a recomputation here disagrees,
#: the ranking being fed to the top-k table is not the one that analysis
#: described, and the command stops rather than printing a table built on it.
TOPK_AP_CHECKS = {"blockscan": 0.388, TOPK_LENGTH_METHOD: 0.492}


def cmd_topk(args: argparse.Namespace) -> int:
    """Pooled top-k evaluation, BlockScan Table 2 style, on the saved predictions.

    Ranks every transaction in one partition by each method's score and counts
    the attacks in the top k. No threshold is involved, so a method with no
    fitted operating point - here, trace length on its own - is directly
    comparable with the three detectors.

    Reads only: the frozen manifest, the split, and the saved per-transaction
    predictions. It trains nothing and fetches nothing.
    """
    import pandas as pd

    from .assemble import attack_classes_of, labels_of
    from . import evaluate as E
    from . import topk as T
    from .splits import load_partitions
    from .tasks import build_tasks, load_vulnerability_index

    cfg, layout = _context(args)
    manifest = _manifest(layout)
    labels = labels_of(manifest)
    classes = attack_classes_of(manifest)
    partitions = load_partitions(layout.splits_dir / "primary_with_validation.parquet")

    index = None
    try:
        rel = _release(cfg, layout, args)
        if rel.has("accepted_attacks"):
            index = load_vulnerability_index(rel.path("accepted_attacks"))
    except TxBenchError as exc:
        log.warning("no release available: %s", redact(exc))
    tasks, _ = build_tasks(manifest, vulnerability_index=index)
    if args.task not in tasks:
        raise TxBenchError(f"unknown task {args.task!r}; have {sorted(tasks)}")
    task = tasks[args.task]

    keys = sorted(k for k in task.eligible if partitions.get(k) == args.partition)
    attacks = [k for k in keys if labels.get(k) == 1]
    negatives = [k for k in keys if labels.get(k) == 0]
    # The ranked population is asserted, not assumed: a top-k table is only
    # comparable across methods if every method ranked the same hashes.
    if args.expect_total is not None and len(keys) != args.expect_total:
        raise TxBenchError(f"expected {args.expect_total} eligible {args.partition} hashes "
                           f"for task {args.task}, found {len(keys)}")
    if args.expect_attacks is not None and len(attacks) != args.expect_attacks:
        raise TxBenchError(f"expected {args.expect_attacks} attacks, found {len(attacks)}")
    if args.expect_negatives is not None and len(negatives) != args.expect_negatives:
        raise TxBenchError(f"expected {args.expect_negatives} negatives, found {len(negatives)}")
    if len(set(keys)) != len(keys):
        raise TxBenchError("the eligible set holds duplicate hashes")
    excluded_classes = sorted({classes[k] for k in task.eligible if k in classes} - {"exploit"})
    leaked = [k for k in keys if classes.get(k) == "key_compromise"]
    if leaked:
        raise TxBenchError(f"{len(leaked)} key compromises are inside the {args.task} task; "
                           "this table is the exploit task only")
    log.info("%s / %s: %d hashes = %d attacks + %d negatives; key compromises excluded",
             args.task, args.partition, len(keys), len(attacks), len(negatives))

    bundles = E.read_predictions(Path(args.predictions), partitions=partitions)
    wanted = {name.split("@")[0]: name for name in bundles}
    methods: dict[str, dict[str, float]] = {}
    provenance: dict[str, Any] = {}
    for short in ("blockscan", "defiguard", "ft_transformer"):
        if short not in wanted:
            raise TxBenchError(f"no predictions for {short} in {args.predictions}")
        bundle = bundles[wanted[short]]
        methods[short] = {k: bundle.predictions[k].oriented_score for k in keys
                          if k in bundle.predictions
                          and bundle.predictions[k].oriented_score is not None}
        directions = {bundle.predictions[k].direction for k in keys
                      if k in bundle.predictions}
        provenance[short] = {"model": wanted[short],
                             "score_direction_recorded": sorted(d for d in directions if d),
                             "oriented": "higher is more attack-like after orientation"}

    # The length baseline is the same `masked_tokens` column the score-composition
    # analysis used - read from the BlockScan run's own output, not recomputed, so
    # the two analyses cannot drift apart.
    length_path = Path(args.length_signal)
    if not length_path.is_file():
        raise TxBenchError(f"no BlockScan prediction parquet at {length_path} to take "
                           "the masked_tokens length signal from")
    frame = pd.read_parquet(length_path)
    if args.length_column not in frame.columns:
        raise TxBenchError(f"{length_path} has no {args.length_column!r} column")
    lengths = {row.tx_key: float(getattr(row, args.length_column))
               for row in frame.itertuples(index=False)
               if row.tx_key in set(keys) and pd.notna(getattr(row, args.length_column))}
    methods[TOPK_LENGTH_METHOD] = lengths
    provenance[TOPK_LENGTH_METHOD] = {
        "source": str(length_path), "column": args.length_column,
        "oriented": "higher is more attack-like (longer trace)",
        "note": "no fitted parameters; the same signal as the score-composition analysis"}

    for name, scores in methods.items():
        if set(scores) != set(keys):
            short = sorted(set(keys) - set(scores))
            raise TxBenchError(f"{name} has finite scores for {len(scores)} of {len(keys)} "
                               f"hashes (first missing: {short[0] if short else 'n/a'})")
        bad = [k for k, v in scores.items() if not math.isfinite(v)]
        if bad:
            raise TxBenchError(f"{name} has {len(bad)} non-finite scores "
                               f"(first: {bad[0]})")
    log.info("all %d methods score exactly the same %d hashes", len(methods), len(keys))

    rankings = {name: T.rank(scores, labels, attack_classes=classes, keys=keys)
                for name, scores in methods.items()}
    aps = {name: T.average_precision(order) for name, order in rankings.items()}
    disagreements = {}
    for name, expected in TOPK_AP_CHECKS.items():
        got = aps.get(name)
        if got is None:
            disagreements[name] = "no AP could be computed"
        elif round(got, 3) != expected:
            disagreements[name] = f"recomputed {got:.6f} (rounds to {round(got, 3)}), "\
                                  f"expected {expected}"
    if disagreements:
        raise TxBenchError(
            "the ranking fed to this table does not reproduce the earlier AP analysis: "
            + "; ".join(f"{k}: {v}" for k, v in disagreements.items())
            + ". Investigate before reporting a top-k table built on it.")
    log.info("AP cross-check passed: %s",
             ", ".join(f"{k}={round(aps[k], 3)}" for k in TOPK_AP_CHECKS))

    cutoffs = [int(k) for k in args.k]
    results: list[Any] = []
    for name in ("blockscan", "defiguard", "ft_transformer", TOPK_LENGTH_METHOD):
        for k in cutoffs:
            results.append(T.top_k(rankings[name], k, method=name))

    print()
    print(f"Pooled top-k evaluation - task {args.task}, {args.partition} partition")
    print(f"{len(keys)} transactions = {len(attacks)} attacks + {len(negatives)} negatives; "
          f"ties broken by ascending tx_hash")
    print()
    print(T.table(results))
    print()
    print("P@k = attacks found / k.  R@k = attacks found / "
          f"{len(attacks)}.  FPR = false alerts / {len(negatives)}.")
    print("tie = attacks reachable at that k over any valid ordering of a tie "
          "straddling the cutoff.")

    top10 = {name: order[:10] for name, order in rankings.items()}
    shared = T.overlap(top10["blockscan"], top10[TOPK_LENGTH_METHOD],
                       b_full=rankings[TOPK_LENGTH_METHOD])
    bs10 = next(r for r in results if r.method == "blockscan" and r.k == 10)
    ln10 = next(r for r in results if r.method == TOPK_LENGTH_METHOD and r.k == 10)
    print()
    print("Top 10: BlockScan against length alone")
    print(f"  BlockScan      {bs10.attacks_found} attacks, {bs10.false_alerts} false alerts "
          f"(P@10 {bs10.precision_at_k:.3f})")
    print(f"  length alone   {ln10.attacks_found} attacks, {ln10.false_alerts} false alerts "
          f"(P@10 {ln10.precision_at_k:.3f})")
    print(f"  shared hashes  {shared['shared']} of 10 (Jaccard {shared['jaccard']:.3f}), "
          f"{shared['shared_attacks']} of them attacks")
    block = shared.get("second_method_tied_block")
    if block:
        print(f"  but length alone cannot order inside its top tie: {block['size']} "
              f"transactions share {block['score']:.6g} "
              f"({block['attacks_in_block']} of them attacks), so its top 10 is drawn "
              "from that block by hash order.")
        print(f"  Against the block rather than its arbitrary members, "
              f"{block['shared_with_block']} of BlockScan's top 10 fall inside it - the "
              "two methods agree on the pool and differ only on ordering within it.")
    if args.rank_correlation:
        from scipy.stats import spearmanr
        rho = spearmanr([r.score for r in sorted(rankings['blockscan'],
                                                 key=lambda r: r.tx_key)],
                        [r.score for r in sorted(rankings[TOPK_LENGTH_METHOD],
                                                 key=lambda r: r.tx_key)]).statistic
        print(f"  Spearman(BlockScan score, masked_tokens) over all {len(keys)} = {rho:.4f}")
        metrics_rho = float(rho)
    else:
        metrics_rho = None

    out_dir = Path(args.out_dir) if args.out_dir else layout.eval_dir / "topk"
    metrics = {
        "task": task.as_dict(),
        "partition": args.partition,
        "population": {"hashes": len(keys), "attacks": len(attacks),
                       "negatives": len(negatives),
                       "attack_classes_present": sorted({classes[k] for k in attacks}),
                       "attack_classes_excluded_from_this_task": excluded_classes},
        "cutoffs": cutoffs,
        "tie_break": "descending score, then ascending tx_hash; labels are not consulted",
        "threshold": None,
        "threshold_note": "top-k needs no decision threshold, so none was used",
        "predictions": str(args.predictions),
        "methods": provenance,
        "average_precision": {k: v for k, v in aps.items()},
        "average_precision_checks": TOPK_AP_CHECKS,
        "average_precision_check_passed": True,
        "results": [r.to_dict() for r in results],
        "top10_blockscan_vs_length": shared,
        "spearman_blockscan_vs_length": metrics_rho,
    }
    written = {"metrics": T.write_json(metrics, out_dir / "topk_metrics.json")}
    selections = [row for r in results for row in T.selection_rows(r)]
    frame = pd.DataFrame(selections)
    out_dir.mkdir(parents=True, exist_ok=True)
    written["selections_csv"] = out_dir / "topk_selections.csv"
    frame.to_csv(written["selections_csv"], index=False)
    written["selections_parquet"] = out_dir / "topk_selections.parquet"
    frame.to_parquet(written["selections_parquet"], index=False)
    print()
    print("Written:")
    for label, path in written.items():
        print(f"  {label:20s} {path}")
    return 0


def _sweep_run_dir(root: Path, seed: int) -> Path:
    return root / f"seed_{seed:02d}"


def _sweep_launch(method: str, seed: int, run_dir: Path, args: argparse.Namespace) -> None:
    """Run one detector at one seed, into its own directory."""
    import subprocess

    base = [sys.executable, "-m", "txbench.cli"]
    if method == "blockscan":
        # Only the test partition is rescored. Each partition is scored in its own
        # process with its own seeding, so the test scores do not depend on whether
        # train and validation were scored alongside them - verified by rerunning
        # seed 42 test-only against the primary artifacts.
        command = base + ["score-blockscan", "--partitions", args.blockscan_partitions,
                          "--seed", str(seed), "--out-dir", str(run_dir)]
    elif method == "defiguard":
        command = base + ["train-defiguard", "--seed", str(seed),
                          "--epochs", str(args.defiguard_epochs),
                          "--patience", str(args.defiguard_patience),
                          "--out-dir", str(run_dir)]
    elif method == "ft_transformer":
        command = base + ["train-ft-transformer", "--views", args.ft_view,
                          "--seed", str(seed), "--epochs", str(args.ft_epochs),
                          "--patience", str(args.ft_patience), "--out-dir", str(run_dir)]
    else:
        raise TxBenchError(f"unknown method {method!r}")
    log.info("seed %d / %s: %s", seed, method, " ".join(command[2:]))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise TxBenchError(f"{method} at seed {seed} failed (exit {completed.returncode}). "
                           f"stderr tail: {completed.stderr[-1200:]}")


def _sweep_read(method: str, run_dir: Path, keys: set[str],
                ft_view: str) -> tuple[dict[str, float], int | None]:
    """One run's per-hash scores and the seed its own process recorded."""
    import pandas as pd

    names = {"blockscan": "predictions_blockscan.parquet",
             "defiguard": "predictions_defiguard.parquet",
             "ft_transformer": f"predictions_ft_transformer_{ft_view.replace('-', '_')}.parquet"}
    path = run_dir / names[method]
    if not path.is_file():
        raise TxBenchError(f"{method}: no predictions at {path}")
    frame = pd.read_parquet(path)
    scored = frame[(frame.status == "scored") & (frame.tx_key.isin(keys))]
    scores = {row.tx_key: float(row.score) for row in scored.itertuples(index=False)
              if pd.notna(row.score)}
    reports = {"blockscan": "blockscan_run.json", "defiguard": "defiguard_run.json",
               "ft_transformer": "ft_transformer_run.json"}
    effective: int | None = None
    report_path = run_dir / reports[method]
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if method == "blockscan":
            block = (report.get("partitions_scored") or {}).get("test") or {}
            effective = block.get("effective_numpy_seed")
        elif method == "defiguard":
            effective = ((report.get("training") or {}).get("effective_seed") or {}
                         ).get("torch_initial_seed")
        else:
            view = (report.get("views") or {}).get(ft_view) or {}
            effective = ((view.get("training") or {}).get("effective_seed") or {}
                         ).get("torch_initial_seed")
    return scores, effective


def cmd_seed_sweep(args: argparse.Namespace) -> int:
    """Repeat the frozen exploit benchmark over a seed range and report the spread.

    Seed 42 is re-run first and checked against the existing primary artifacts.
    Only if it reproduces their top-k counts are the remaining seeds launched:
    a configuration that no longer reproduces the published numbers would be a
    different experiment, and averaging it with them would hide that.

    Every seed that runs is reported, including the ones that came out badly.
    """
    import pandas as pd

    from .assemble import attack_classes_of, incidents_of, labels_of
    from . import evaluate as E
    from . import seedsweep as S
    from .splits import load_partitions
    from .tasks import build_tasks, load_vulnerability_index

    cfg, layout = _context(args)
    manifest = _manifest(layout)
    labels = labels_of(manifest)
    incidents = incidents_of(manifest)
    classes = attack_classes_of(manifest)
    partitions = load_partitions(layout.splits_dir / "primary_with_validation.parquet")

    index = None
    try:
        rel = _release(cfg, layout, args)
        if rel.has("accepted_attacks"):
            index = load_vulnerability_index(rel.path("accepted_attacks"))
    except TxBenchError as exc:
        log.warning("no release available: %s", redact(exc))
    tasks, _ = build_tasks(manifest, vulnerability_index=index)
    task = tasks[args.task]
    keys = sorted(k for k in task.eligible if partitions.get(k) == args.partition)
    attacks = [k for k in keys if labels.get(k) == 1]
    if args.expect_total is not None and len(keys) != args.expect_total:
        raise TxBenchError(f"expected {args.expect_total} {args.partition} hashes, "
                           f"found {len(keys)}")
    cutoffs = sorted({int(k) for k in args.k})
    log.info("sweep population: %d hashes = %d attacks + %d negatives",
             len(keys), len(attacks), len(keys) - len(attacks))

    seeds = _parse_seeds(args.seeds)
    if S.PRIMARY_SEED not in seeds:
        raise TxBenchError(f"the sweep must include seed {S.PRIMARY_SEED}: it is the "
                           "reproduction reference for the primary artifacts")

    # The existing primary run, measured the same way, is the reference.
    primary_bundles = E.read_predictions(Path(args.primary_predictions), partitions=partitions)
    primary: dict[str, Any] = {}
    for name, bundle in primary_bundles.items():
        short = name.split("@")[0]
        if short not in S.SWEPT_METHODS:
            continue
        scores = {k: bundle.predictions[k].oriented_score for k in keys
                  if k in bundle.predictions
                  and bundle.predictions[k].oriented_score is not None}
        if set(scores) != set(keys):
            raise TxBenchError(f"the primary {short} predictions cover {len(scores)} of "
                               f"{len(keys)} hashes")
        primary[short] = S.measure(scores, labels, method=short, seed=S.PRIMARY_SEED,
                                   effective_seed=None, keys=keys, cutoffs=cutoffs,
                                   incidents=incidents, attack_classes=classes)
    missing = [m for m in S.SWEPT_METHODS if m not in primary]
    if missing:
        raise TxBenchError(f"the primary predictions lack {missing}")

    sweep_root = Path(args.out_dir) if args.out_dir else layout.eval_dir / "seeds"
    sweep_root.mkdir(parents=True, exist_ok=True)
    runs: list[Any] = []
    reproduction: list[dict[str, Any]] = []

    ordered_seeds = [S.PRIMARY_SEED] + [s for s in seeds if s != S.PRIMARY_SEED]
    for seed in ordered_seeds:
        for method in S.SWEPT_METHODS:
            run_dir = _sweep_run_dir(sweep_root, seed) / method
            done = (run_dir / "predictions_blockscan.parquet").is_file() if \
                method == "blockscan" else (
                    run_dir / ("predictions_defiguard.parquet" if method == "defiguard"
                               else f"predictions_ft_transformer_"
                                    f"{args.ft_view.replace('-', '_')}.parquet")).is_file()
            if done and not args.force:
                log.info("seed %d / %s: already present, reusing", seed, method)
            else:
                _sweep_launch(method, seed, run_dir, args)
            scores, effective = _sweep_read(method, run_dir, set(keys), args.ft_view)
            if set(scores) != set(keys):
                raise TxBenchError(
                    f"{method} at seed {seed} scored {len(scores)} of {len(keys)} hashes; "
                    "every run must cover exactly the same hashes")
            if effective is not None and int(effective) != seed:
                raise TxBenchError(
                    f"{method} at seed {seed} recorded an effective seed of {effective}: "
                    "the seed did not take effect inside the process, so the run is not "
                    "the one requested")
            runs.append(S.measure(scores, labels, method=method, seed=seed,
                                  effective_seed=effective, keys=keys, cutoffs=cutoffs,
                                  incidents=incidents, attack_classes=classes))
        if seed == S.PRIMARY_SEED:
            for method in S.SWEPT_METHODS:
                rerun = next(r for r in runs if r.method == method and r.seed == seed)
                check = S.reproduces(rerun, primary[method], cutoffs=cutoffs)
                reproduction.append(check)
                log.info("seed 42 %s reproduces the primary artifacts: %s",
                         method, check["reproduced"])
            failed = [c for c in reproduction if not c["reproduced"]]
            if failed and not args.allow_reproduction_mismatch:
                raise TxBenchError(
                    "seed 42 does not reproduce the primary top-k counts: "
                    + "; ".join(f"{c['method']} {c['checks']}" for c in failed)
                    + ". The sweep is stopped rather than averaging a different "
                      "configuration with the published one.")

    summary = S.summarise(runs, k=args.summary_k)
    heads = [S.head_to_head(runs, challenger=m, baseline="blockscan", k=args.summary_k)
             for m in ("defiguard", "ft_transformer")]

    print()
    print(f"Seed sweep - task {args.task}, {args.partition} partition, "
          f"{len(keys)} hashes ({len(attacks)} attacks)")
    print(f"seeds {min(seeds)}-{max(seeds)} ({len(seeds)} runs per method); every run kept")
    print()
    print(S.table(runs, cutoffs))
    print()
    print(f"Primary top-{args.summary_k} result across all {len(seeds)} seeds")
    for method in S.SWEPT_METHODS:
        block = summary[method]
        print(f"  {method:15s} {block['attacks_at_k']}  mean {block['mean']:.1f}  "
              f"range {block['min']}-{block['max']}  sd {block['stdev']:.2f}  "
              f"AP {block['average_precision_mean']:.3f} "
              f"[{block['average_precision_min']:.3f}-{block['average_precision_max']:.3f}]")
    print()
    print(f"Head to head at k={args.summary_k}, per seed")
    for head in heads:
        print(f"  {head['challenger']:15s} vs blockscan: "
              f"{head['wins']}W / {head['ties']}T / {head['losses']}L of "
              f"{head['seeds_compared']}  (win rate {head['win_rate']:.0%}, "
              f"mean margin {head['mean_margin']:+.1f}, "
              f"range {head['margin_range'][0]:+d} to {head['margin_range'][1]:+d})")
    ties = [(r.method, r.seed, t) for r in runs for t in r.ties_crossing_cutoff]
    print()
    if ties:
        print(f"Ties crossing a cutoff ({len(ties)}):")
        for method, seed, tie in ties:
            print(f"  {method} seed {seed} k={tie['k']}: score {tie['cutoff_score']:.6g}, "
                  f"{tie['tied_at_cutoff']} tied for {tie['slots_remaining']} slot(s), "
                  f"realised {tie['attacks_found']} in "
                  f"[{tie['reachable'][0]}, {tie['reachable'][1]}]")
    else:
        print("No tie crossed any cutoff in any run.")

    payload = {
        "task": task.as_dict(), "partition": args.partition,
        "population": {"hashes": len(keys), "attacks": len(attacks),
                       "negatives": len(keys) - len(attacks)},
        "seeds": seeds, "cutoffs": cutoffs, "summary_k": args.summary_k,
        "every_seed_kept": True,
        "primary_seed": S.PRIMARY_SEED,
        "primary_predictions": str(args.primary_predictions),
        "reproduction_check": reproduction,
        "configuration": {
            "blockscan": {"partitions": args.blockscan_partitions,
                          "note": "the authors' checkpoint, rescored; nothing is retrained"},
            "defiguard": {"epochs": args.defiguard_epochs,
                          "patience": args.defiguard_patience},
            "ft_transformer": {"view": args.ft_view, "epochs": args.ft_epochs,
                               "patience": args.ft_patience,
                               "note": "the view selected on validation at seed 42, held "
                                       "fixed across seeds so the sweep varies only the seed"},
        },
        "runs": [r.to_dict() for r in runs],
        "summary": summary,
        "head_to_head": heads,
        "ties_crossing_cutoff": [{"method": m, "seed": s, **t} for m, s, t in ties],
    }
    written = {"metrics": S.write_json(payload, sweep_root / "seed_sweep.json")}
    rows = [{"method": r.method, "seed": r.seed, "effective_seed": r.effective_seed,
             "k": k, **{kk: vv for kk, vv in block.items()}}
            for r in runs for k, block in sorted(r.at_k.items())]
    frame = pd.DataFrame(rows)
    written["metrics_csv"] = sweep_root / "seed_sweep_metrics.csv"
    frame.to_csv(written["metrics_csv"], index=False)
    # Per-hash scores for every run, so any figure here can be recomputed.
    per_hash = []
    for r in runs:
        run_dir = _sweep_run_dir(sweep_root, r.seed) / r.method
        scores, _ = _sweep_read(r.method, run_dir, set(keys), args.ft_view)
        per_hash += [{"method": r.method, "seed": r.seed, "tx_key": k,
                      "tx_hash": k.split(":", 1)[1], "score": v,
                      "label": labels[k], "attack_class": classes.get(k),
                      "group_incident": incidents.get(k)}
                     for k, v in sorted(scores.items())]
    scores_frame = pd.DataFrame(per_hash)
    written["per_hash_parquet"] = sweep_root / "seed_sweep_scores.parquet"
    scores_frame.to_parquet(written["per_hash_parquet"], index=False)
    written["per_hash_csv"] = sweep_root / "seed_sweep_scores.csv"
    scores_frame.to_csv(written["per_hash_csv"], index=False)
    print()
    print("Written:")
    for label, path in written.items():
        print(f"  {label:20s} {path}")
    return 0


def _parse_seeds(spec: str) -> list[int]:
    """`42-51`, `42,43,44` or a mix."""
    out: list[int] = []
    for piece in str(spec).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece.lstrip("-"):
            low, high = piece.split("-", 1)
            if int(low) > int(high):
                raise TxBenchError(f"seed range {piece!r} runs backwards")
            out += list(range(int(low), int(high) + 1))
        else:
            out.append(int(piece))
    if not out:
        raise TxBenchError("no seeds given")
    if len(set(out)) != len(out):
        raise TxBenchError(f"duplicate seeds in {spec!r}")
    return sorted(out)


def cmd_compare_detectors(args: argparse.Namespace) -> int:
    """Put every model's per-transaction answer in one place, on the same hashes.

    One long table (one row per transaction per model, with the status that
    model gave) and one wide table (one score column per model, aligned by
    transaction) - so a comparison cannot quietly be made over different sets
    of hashes. Coverage is reported beside the scores, never folded into them.
    """
    import pandas as pd

    from .assemble import attack_classes_of, labels_of
    from .models import merge_predictions_csv, write_json
    from .splits import load_partitions

    cfg, layout = _context(args)
    manifest = _manifest(layout)
    labels = labels_of(manifest)
    classes = attack_classes_of(manifest)
    partitions = load_partitions(layout.splits_dir / "primary_with_validation.parquet")

    detectors = layout.eval_dir / "detectors"
    sources = [Path(p) for p in args.predictions] if args.predictions else sorted(
        detectors.glob("predictions_*.csv"))
    sources = [p for p in sources if p.name not in ("predictions_all.csv",)
               and not p.name.startswith("predictions_ft_transformer_")]
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise TxBenchError(f"no prediction file(s): {missing}")
    if not sources:
        raise TxBenchError("no prediction files to compare. Run the three model commands first.")

    out_dir = layout.eval_dir / "predictions"
    merged_csv = merge_predictions_csv(sources, out_dir / "predictions_all.csv")

    long_rows = []
    for path in sources:
        parquet = path.with_suffix(".parquet")
        frame = pd.read_parquet(parquet) if parquet.is_file() else pd.read_csv(path)
        frame = frame.copy()
        frame["source_file"] = path.name
        long_rows.append(frame)
    long = pd.concat(long_rows, ignore_index=True)
    long["partition"] = long["tx_key"].map(partitions)
    long["label"] = long["tx_key"].map(labels)
    long["attack_class"] = long["tx_key"].map(classes)
    long.to_parquet(out_dir / "per_transaction.parquet", index=False)

    scored = long[long["status"] == "scored"]
    wide = scored.pivot_table(index="tx_key", columns="model", values="score", aggfunc="first")
    wide = wide.reset_index()
    wide["partition"] = wide["tx_key"].map(partitions)
    wide["label"] = wide["tx_key"].map(labels)
    wide["attack_class"] = wide["tx_key"].map(classes)
    wide.to_parquet(out_dir / "per_transaction_scores.parquet", index=False)

    models = sorted(long["model"].astype(str).unique())
    coverage = {}
    for model in models:
        rows = long[long["model"] == model]
        coverage[model] = {
            "rows": int(len(rows)),
            "versions": sorted(rows["version"].astype(str).unique()),
            "by_status": {k: int(v) for k, v in
                          rows["status"].value_counts().to_dict().items()},
            "scored_by_partition": {k: int(v) for k, v in
                                    rows[rows["status"] == "scored"]["partition"]
                                    .value_counts().to_dict().items()},
            "unscored": [{"tx_key": r.tx_key, "partition": r.partition, "status": r.status,
                          "note": r.note}
                         for r in rows[rows["status"] != "scored"].itertuples(index=False)],
        }
    score_columns = [m for m in models if m in wide.columns]
    complete = wide.dropna(subset=score_columns)

    # Reproducibility in one place: each model's variant, its training or
    # checkpoint configuration, and the seed that produced these scores.
    runs: dict[str, Any] = {}
    for name, filename in (("blockscan", "blockscan_run.json"),
                           ("defiguard", "defiguard_run.json"),
                           ("ft_transformer", "ft_transformer_run.json")):
        path = detectors / filename
        if not path.is_file():
            continue
        block = json.loads(path.read_text(encoding="utf-8"))
        entry: dict[str, Any] = {"report": str(path.relative_to(layout.root)),
                                 "variant": block.get("variant"),
                                 "evidence_level": block.get("evidence_level")}
        if name == "blockscan":
            entry.update({
                "version": block.get("version"),
                "seed": block.get("seed"),
                "scorer_settings": block.get("scorer_settings"),
                "scorer_python": block.get("scorer_python"),
                "assets": {k: {"path": v.get("path"), "sha256": v.get("sha256"),
                               "entries": v.get("entries"), "weights": v.get("weights")}
                           for k, v in (block.get("assets", {}).get("present") or {}).items()},
                "model": (block.get("smoke_test") or {}).get("model"),
                "environment": (block.get("smoke_test") or {}).get("environment"),
                "truncation": {partition: slot.get("truncation")
                               for partition, slot in
                               (block.get("partitions_scored") or {}).items()},
            })
        elif name == "defiguard":
            entry.update({
                "version": block.get("version"),
                "training_config": (block.get("training") or {}).get("config"),
                "selected_epoch": (block.get("training") or {}).get("selected_epoch"),
                "validation_average_precision": (block.get("training") or {}).get(
                    "validation_ap"),
                "node_features": block.get("node_features"),
                "omitted_node_features": block.get("omitted_node_features"),
                "excluded": block.get("excluded"),
                "checkpoint": "data/evaluation/detectors/defiguard_gnn.pt",
            })
        else:
            primary = block.get("primary_view")
            view = (block.get("views") or {}).get(primary) or {}
            entry.update({
                "version": block.get("primary_version"),
                "primary_view": primary,
                "primary_selected_on": block.get("primary_selected_on"),
                "training_config": (view.get("training") or {}).get("config"),
                "selected_epoch": (view.get("training") or {}).get("selected_epoch"),
                "validation_average_precision_by_view": block.get(
                    "validation_average_precision_by_view"),
                "feature_selection": view.get("selection"),
                "checkpoint": f"data/evaluation/detectors/ft_transformer_"
                              f"{str(primary).replace('-', '_')}.pt",
            })
        runs[name] = entry

    report = {
        "models": models,
        "runs": runs,
        "transactions_in_split": len(partitions),
        "coverage": coverage,
        "scored_by_every_model": int(len(complete)),
        "scored_by_every_model_by_partition": {
            k: int(v) for k, v in complete["partition"].value_counts().to_dict().items()},
        "note": "a transaction every model scored is the only fair ground for a paired "
                "comparison; the coverage block above says what each model left unscored, "
                "and those rows stay in the denominator of its own metrics",
        "artifacts": {
            "predictions_csv": str(merged_csv),
            "per_transaction_parquet": str(out_dir / "per_transaction.parquet"),
            "per_transaction_scores_parquet": str(out_dir / "per_transaction_scores.parquet"),
        },
    }
    artifact = write_json(report, out_dir / "comparison.json")
    _emit({"models": models, "scored_by_every_model": report["scored_by_every_model"],
           "coverage": {m: coverage[m]["by_status"] for m in models},
           "artifacts": report["artifacts"], "report": str(artifact)})
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .report import write_status_report
    cfg, layout = _context(args)
    written = write_status_report(layout, cfg)
    _emit({"artifacts": {k: str(v) for k, v in written.items()}})
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    """Print the rebuild command sequence, in order."""
    from .report import REBUILD_STEPS
    print("\n".join(REBUILD_STEPS))
    return 0


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"txbench {__version__}")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("-q", "--quiet", action="store_true")
    parser.add_argument("--root", help="repository root (default: discovered)")
    parser.add_argument("--config", help="TOML config file (default: config/benchmark.toml)")
    sub = parser.add_subparsers(dest="command", required=True)

    def release_args(p):
        p.add_argument("--release-dir", help="the upstream TxBench release directory")
        p.add_argument("--release-id", help="override the recorded release identifier")
        p.add_argument("--doi", help="the release DOI, recorded as provenance; "
                                    "defaults to cohort.release_doi in the config")
        p.add_argument("--skip-checksums", action="store_true",
                       help="skip SHA256SUMS verification (not recommended)")

    p = sub.add_parser("fetch-release", help="download, checksum and extract the release")
    p.add_argument("--record", default="23758032", help="Zenodo record id")
    p.add_argument("--md5", help="expected archive md5 (recorded and verified when given)")
    p.add_argument("--no-keep-archive", action="store_true")
    p.set_defaults(func=cmd_fetch_release)

    p = sub.add_parser("import-txbench", help="import labels and freeze the cohort")
    release_args(p)
    p.add_argument("--chain-id", type=int, default=None)
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("collect", help="inventory, price and fetch the missing raw data")
    release_args(p)
    p.add_argument("--dry-run", action="store_true", help="plan and price; make no calls")
    p.add_argument("--pilot", type=int, help="restrict to a deterministic pilot of N txs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--provider", default="alchemy")
    p.add_argument("--fallback", default=None)
    p.add_argument("--resume", action="store_true",
                   help="reuse every completed request (always true; kept for clarity)")
    p.add_argument("--no-probe", action="store_true", help="skip the provider capability probe")
    p.add_argument("--verify-upstream", action="store_true",
                   help="re-read fields the upstream record already supplies, to cross-check")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("audit-inputs", help="per-detector field coverage")
    release_args(p)
    p.add_argument("--pilot", action="store_true", help="audit the recorded pilot only")
    p.add_argument("--detector", action="append", help="limit to a detector (repeatable)")
    p.add_argument("--blockscan-repo", help="path to the pinned BlockScan checkout")
    p.add_argument("--ignore-recovered-traces", action="store_true", help="judge the release's records as supplied, without applying the call trees `bench enrich-traces` recovered (the before state, for comparison)")
    p.set_defaults(func=cmd_audit_inputs)

    p = sub.add_parser("validate-adapters", help="full-cohort detector input validation")
    release_args(p)
    p.add_argument("--detector", action="append")
    p.add_argument("--blockscan-repo")
    p.add_argument("--ignore-recovered-traces", action="store_true", help="judge the release's records as supplied, without applying the call trees `bench enrich-traces` recovered (the before state, for comparison)")
    p.set_defaults(func=cmd_validate_adapters)

    p = sub.add_parser("backfill-init", help="fetch creation init bytes for the frames "
                                            "that still lack them")
    release_args(p)
    p.add_argument("--table", default=None, help="override the configured traces table")
    p.add_argument("--project", default=None, help="the Google Cloud project to bill")
    p.add_argument("--budget-bytes", type=int, default=None,
                   help="total bytes this run may bill; enforced per job")
    p.add_argument("--dry-run", action="store_true", help="price the scan and stop")
    p.set_defaults(func=cmd_backfill_init, ignore_recovered_traces=False, pilot=False,
                   detector=None, blockscan_repo=None)

    p = sub.add_parser("enrich-traces", help="recover call trees from BigQuery for the "
                                            "transactions a missing call path blocks")
    p.add_argument("--detector", default="defiguard",
                   help="whose blocked transactions to recover (default: defiguard)")
    p.add_argument("--scope", default="cohort", choices=("cohort", "pilot"),
                   help="which coverage output to read the blocked hashes from")
    p.add_argument("--select", default="blocked", choices=("blocked", "all"),
                   help="which of that output's transactions to enrich: the detector's "
                        "blocked ones (the default, and the pilot), or every transaction "
                        "in the frozen set")
    p.add_argument("--expect", type=int, default=None,
                   help="the number of blocked hashes expected; a difference is explained "
                        "and nothing is queried")
    p.add_argument("--allow-count-mismatch", action="store_true",
                   help="proceed even when --expect disagrees with the coverage output")
    p.add_argument("--table", default=None, help="override the configured traces table")
    p.add_argument("--project", default=None, help="the Google Cloud project to bill")
    p.add_argument("--budget-bytes", type=int, default=None,
                   help="total bytes this run may bill; enforced per job through "
                        "maximum_bytes_billed")
    p.add_argument("--dry-run", action="store_true",
                   help="price the scan and stop")
    p.add_argument("--rebuild-only", action="store_true",
                   help="rebuild from cached rows without contacting BigQuery")
    p.add_argument("--blockscan-repo", default=None,
                   help="path to the pinned BlockScan checkout for the preprocessor run")
    p.set_defaults(func=cmd_enrich_traces)

    p = sub.add_parser("build-features", help="build one feature view")
    release_args(p)
    p.add_argument("--view", required=True, choices=["tx-receipt", "full-calls-receipt"])
    p.add_argument("--split", help="split whose train rows fit the vocabulary")
    p.set_defaults(func=cmd_build_features)

    p = sub.add_parser("verify-splits", help="audit the upstream splits, add a validation part")
    p.set_defaults(func=cmd_verify_splits)

    p = sub.add_parser("build-tasks", help="freeze the exploit / key-compromise / "
                                           "price-manipulation task definitions")
    release_args(p)
    p.set_defaults(func=cmd_build_tasks)

    p = sub.add_parser("evaluate", help="score saved predictions")
    release_args(p)
    p.add_argument("--predictions", required=True, action="append",
                   help="a predictions CSV; repeatable, and several are merged before scoring")
    p.add_argument("--task", default="exploit",
                   choices=["exploit", "key_compromise", "price_manipulation"])
    p.add_argument("--fixture-mode", action="store_true",
                   help="mark every number as fixture-derived")
    p.add_argument("--strict-eligible", action="store_true",
                   help="refuse prediction rows outside the task's eligible hashes")
    p.add_argument("--threshold-rule", default="max_f1",
                   choices=["max_f1", "target_fp_per_1k"])
    p.add_argument("--target-fp-per-1k", type=float, default=None)
    p.add_argument("--split", default="temporal", help="split passed to the upstream evaluator")
    p.add_argument("--track", default="exploit", help="track passed to the upstream evaluator")
    p.add_argument("--base-rate", type=float, default=None)
    p.add_argument("--tolerance", type=float, default=1e-9)
    p.add_argument("--out", default=None,
                   help="where to write the evaluation artifact "
                        "(default: data/evaluation/evaluation.json)")
    p.add_argument("--thresholds-from", default=None,
                   help="carry each model's frozen threshold from an earlier evaluation "
                        "artifact instead of choosing one on this task's validation rows")
    p.add_argument("--no-parity", action="store_true",
                   help="skip the upstream-evaluator parity check")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("score-blockscan",
                       help="score the frozen set with BlockScan's own model and scorer")
    release_args(p)
    p.add_argument("--blockscan-repo", default=None, help="the pinned BlockScan checkout")
    p.add_argument("--scorer-python", default=None,
                   help="an interpreter with the authors' pinned dependency set "
                        "(torch 2.1.0, transformers 4.32.0); defaults to BLOCKSCAN_PYTHON")
    p.add_argument("--partitions", default="train,validation,test")
    p.add_argument("--smoke-partition", default="train")
    p.add_argument("--smoke-only", action="store_true",
                   help="run the smoke test and stop")
    p.add_argument("--work-dir", default=None)
    p.add_argument("--limit", type=int, default=None, help="score only the first N (debugging)")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--mask-percent", type=int, default=15)
    p.add_argument("--top-s", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore-recovered-traces", action="store_true",
                   help="score the release's records as supplied, without the recovered "
                        "call trees (the before state, for comparison)")
    p.add_argument("--out-dir", help="write this run's artifacts here instead of\n                        data/evaluation/detectors (used by the seed sweep)")
    p.set_defaults(func=cmd_score_blockscan)

    p = sub.add_parser("train-defiguard",
                       help="train the paper-based DeFiGuard GNN on the frozen split")
    release_args(p)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ignore-recovered-traces", action="store_true")
    p.add_argument("--out-dir", help="write this run's artifacts here instead of\n                        data/evaluation/detectors (used by the seed sweep)")
    p.set_defaults(func=cmd_train_defiguard)

    p = sub.add_parser("train-ft-transformer",
                       help="train the FT-Transformer on this project's feature views")
    p.add_argument("--views", default="tx-receipt,full-calls-receipt")
    p.add_argument("--feature-split", default="temporal")
    p.add_argument("--min-document-frequency", type=int, default=5,
                   help="drop vocabulary tokens below this training document frequency; the "
                        "frequency table was fitted on training rows only")
    p.add_argument("--d-token", type=int, default=16)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", help="write this run's artifacts here instead of\n                        data/evaluation/detectors (used by the seed sweep)")
    p.set_defaults(func=cmd_train_ft_transformer)

    p = sub.add_parser("compare-detectors",
                       help="merge every model's predictions and align them per transaction")
    p.add_argument("--predictions", action="append", default=None,
                   help="a predictions CSV; repeatable (default: every one written by the "
                        "model commands)")
    p.set_defaults(func=cmd_compare_detectors)

    p = sub.add_parser("analyze-misses",
                       help="incident concentration and a shape-matched contrast for one "
                            "attack class, at the frozen thresholds")
    p.add_argument("--attack-class", default="key_compromise")
    p.add_argument("--partition", default="test")
    p.add_argument("--view", default="full-calls-receipt")
    p.add_argument("--predictions", default="data/evaluation/predictions/predictions_all.csv")
    p.add_argument("--thresholds-from", default="data/evaluation/evaluation_exploit.json",
                   help="evaluation artifact whose frozen thresholds are applied unchanged")
    p.add_argument("--out")
    p.set_defaults(func=cmd_analyze_misses)

    p = sub.add_parser("topk",
                       help="pooled top-k evaluation (BlockScan Table 2 style) over the "
                            "saved predictions; no threshold involved")
    p.add_argument("--task", default="exploit")
    p.add_argument("--partition", default="test")
    p.add_argument("--k", nargs="+", default=["5", "10", "15"])
    p.add_argument("--predictions", default="data/evaluation/predictions/predictions_all.csv")
    p.add_argument("--length-signal",
                   default="data/evaluation/detectors/predictions_blockscan.parquet",
                   help="where the masked_tokens length baseline is read from")
    p.add_argument("--length-column", default="masked_tokens")
    p.add_argument("--expect-total", type=int, default=741)
    p.add_argument("--expect-attacks", type=int, default=129)
    p.add_argument("--expect-negatives", type=int, default=612)
    p.add_argument("--out-dir")
    p.add_argument("--no-rank-correlation", dest="rank_correlation", action="store_false",
                   help="skip the Spearman check between the BlockScan score and length")
    p.set_defaults(func=cmd_topk, rank_correlation=True)

    p = sub.add_parser("seed-sweep",
                       help="repeat the frozen exploit benchmark over a seed range; every "
                            "seed is kept and seed 42 must reproduce the primary artifacts")
    release_args(p)
    p.add_argument("--seeds", default="42-51")
    p.add_argument("--task", default="exploit")
    p.add_argument("--partition", default="test")
    p.add_argument("--k", nargs="+", default=["5", "10", "15"])
    p.add_argument("--summary-k", type=int, default=10)
    p.add_argument("--expect-total", type=int, default=741)
    p.add_argument("--primary-predictions",
                   default="data/evaluation/predictions/predictions_all.csv")
    p.add_argument("--blockscan-partitions", default="test")
    p.add_argument("--defiguard-epochs", type=int, default=300)
    p.add_argument("--defiguard-patience", type=int, default=30)
    p.add_argument("--ft-view", default="tx-receipt")
    p.add_argument("--ft-epochs", type=int, default=250)
    p.add_argument("--ft-patience", type=int, default=30)
    p.add_argument("--out-dir")
    p.add_argument("--force", action="store_true",
                   help="re-run a (method, seed) whose artifacts are already present")
    p.add_argument("--allow-reproduction-mismatch", action="store_true",
                   help="continue even if seed 42 no longer reproduces the primary counts")
    p.set_defaults(func=cmd_seed_sweep)

    p = sub.add_parser("report", help="assemble the status report")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("rebuild", help="print the rebuild command sequence")
    p.set_defaults(func=cmd_rebuild)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(-1 if args.quiet else args.verbose)
    try:
        return int(args.func(args) or 0)
    except TxBenchError as exc:
        print(f"error: {redact(exc)}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
