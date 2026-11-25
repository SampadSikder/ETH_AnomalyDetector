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
    return load_release(directory, release_id=getattr(args, "release_id", None),
                        doi=getattr(args, "doi", None),
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
    """Normalised records for the cohort (or a subset), upstream first."""
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
    return manifest, rel, list(records(manifest, traces_dir=traces_dir, store=store,
                                       trace_status=statuses, keys=keys))


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
    first = next((tx for _, tx in assembled if tx is not None), None)
    if first is not None and repo.is_dir():
        smoke = run_smoke(first, repo)
    out_dir = layout.adapters_dir / ("pilot" if args.pilot else "cohort")
    written = write_audit(checks, out_dir, labels=labels_of(manifest),
                          extras={"scope": scope, "blockscan_smoke_test": smoke,
                                  "records_assembled": len(assembled),
                                  "records_missing": sum(1 for _, tx in assembled if tx is None)})
    from .adapters import coverage_table
    _emit({"scope": scope, "coverage": coverage_table(checks, labels_of(manifest)),
           "blockscan_smoke_test": smoke,
           "artifacts": {k: str(v) for k, v in written.items()}})
    return 0


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
    from .assemble import incidents_of, labels_of
    from . import evaluate as E
    from .splits import TEST, VALIDATION, load_partitions
    from .tasks import TASK_EXPLOIT, build_tasks, load_vulnerability_index
    cfg, layout = _context(args)
    manifest = _manifest(layout)
    labels = labels_of(manifest)
    incidents = incidents_of(manifest)

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
    bundles = E.read_predictions(Path(args.predictions), partitions=partitions,
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
    thresholds: dict[str, float | None] = {}
    for name, bundle in bundles.items():
        completeness = E.check_completeness(bundle, eligible)
        chosen = E.select_threshold(bundle, labels, validation_keys, rule=args.threshold_rule,
                                    target_fp_per_1k=args.target_fp_per_1k)
        thresholds[name] = chosen.get("threshold")
        results["models"][name] = {
            "completeness": completeness,
            "threshold_selection": chosen,
            "validation": E.score_partition(bundle, labels, validation_keys,
                                            threshold=chosen.get("threshold"),
                                            incidents=incidents, partition="validation"),
            "test": E.score_partition(bundle, labels, test_keys,
                                      threshold=chosen.get("threshold"),
                                      incidents=incidents, partition="test"),
        }
    if len(bundles) > 1:
        results["paired_comparison"] = E.paired_comparison(
            results["models"], bundles, labels, test_keys, thresholds=thresholds,
            incidents=incidents)

    parity: dict[str, Any] = {"status": "unverified",
                              "reason": "no pinned upstream evaluator was available"}
    if rel is not None and rel.has("evaluator") and rel.has("combined"):
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

    out = E.write_json(results, layout.eval_dir / "evaluation.json")
    _emit({"task": args.task, "models": sorted(bundles),
           "parity_status": parity.get("status"),
           "test": {k: {"recall": v["test"]["recall"], "precision": v["test"]["precision"],
                        "average_precision": v["test"]["average_precision"],
                        "fp_per_1000_negatives": v["test"]["false_positives_per_1000_negatives"]}
                    for k, v in results["models"].items()},
           "artifact": str(out)})
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
        p.add_argument("--doi", help="the release DOI, recorded as provenance")
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
    p.set_defaults(func=cmd_audit_inputs)

    p = sub.add_parser("validate-adapters", help="full-cohort detector input validation")
    release_args(p)
    p.add_argument("--detector", action="append")
    p.add_argument("--blockscan-repo")
    p.set_defaults(func=cmd_validate_adapters)

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
    p.add_argument("--predictions", required=True)
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
    p.set_defaults(func=cmd_evaluate)

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
