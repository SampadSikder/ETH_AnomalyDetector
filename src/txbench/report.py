"""Reports: the cohort report, and the status report that names every artefact."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .credentials import credential_status
from .logs import get_logger

log = get_logger("report")

REBUILD_STEPS = [
    "# Rebuild, in order (all commands run inside the project environment):",
    "conda run -n txbench bench fetch-release                       # optional: pull the pinned release",
    "conda run -n txbench bench import-txbench --release-dir data/upstream/txbench --chain-id 1",
    "# optional: the pinned BlockScan checkout, for its preprocessing smoke test",
    "git clone https://github.com/nuwuxian/BlockScan.git data/upstream/detectors/blockscan && \\",
    "  git -C data/upstream/detectors/blockscan checkout 4556cd209c968ed60e8d15b245f8d53e3f725534",
    "conda run -n txbench bench collect --dry-run",
    "conda run -n txbench bench collect --pilot 20 --seed 42 --provider alchemy --fallback drpc",
    "conda run -n txbench bench audit-inputs --pilot",
    "conda run -n txbench bench collect --resume --provider alchemy --fallback drpc",
    "conda run -n txbench bench verify-splits",
    "conda run -n txbench bench build-features --view tx-receipt",
    "conda run -n txbench bench build-features --view full-calls-receipt",
    "conda run -n txbench bench build-tasks",
    "conda run -n txbench bench validate-adapters",
    "conda run -n txbench bench evaluate --predictions fixtures/example_predictions.csv --fixture-mode",
    "conda run -n txbench bench report",
]


def _table(rows: list[list[Any]], header: list[str]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        out.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(out)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def write_cohort_report(result, path: Path) -> Path:
    """The cohort report: row counts before and after every filter."""
    meta = result.meta
    lines = [
        "# Cohort report",
        "",
        f"- generated: {meta['generated_utc']}",
        f"- release: `{meta['release']['release_id']}`"
        + (f" (DOI {meta['release']['doi']})" if meta["release"].get("doi") else ""),
        f"- fixture data: **{meta['is_fixture']}**",
        f"- chain id: {meta['chain_id']}",
        f"- rows: **{meta['rows']}**",
        f"- cohort digest (sorted {', '.join(meta['cohort_digest_fields'])}): "
        f"`{meta['cohort_sha256']}`",
        "",
        "## Filters",
        "",
        _table([[s["filter"], s["rows_in"], s["rows_out"], s["dropped"]]
                for s in meta["filters"]],
               ["filter", "rows in", "rows out", "dropped"]),
        "",
        "## Labels",
        "",
        _table([[k, v] for k, v in sorted(meta["label_counts"].items())], ["label", "rows"]),
        "",
        "### Attack classes",
        "",
        _table([[k, v] for k, v in sorted(meta["attack_class_counts"].items())],
               ["attack_class", "rows"]),
        "",
        "### Negative categories",
        "",
        _table([[k, v] for k, v in sorted(meta["benign_category_counts"].items())],
               ["category", "rows"]),
        "",
        "## Quarantine",
        "",
        f"{meta['quarantined_rows']} row(s) quarantined. An unknown label, a failed "
        "validation or a conflicting duplicate is never treated as benign.",
        "",
        _table([[k, v] for k, v in sorted(meta["quarantine_reasons"].items())],
               ["reason", "rows"]) if meta["quarantine_reasons"] else "_none_",
        "",
        "## Upstream splits preserved",
        "",
        ", ".join(f"`{c}`" for c in meta["upstream_split_columns"]) or "_none_",
        "",
        "## Source files and checksums",
        "",
        _table([[role, meta["release"]["files"].get(role, ""),
                 meta["release"]["checksum_status"].get(role, ""),
                 (meta["release"]["sha256"].get(role) or "")[:16]]
                for role in sorted(meta["release"]["files"])],
               ["role", "path", "checksum", "sha256 (short)"]),
        "",
        "## Licence / redistribution",
        "",
        "The release's own licence files are recorded in "
        "`data/manifest/release_provenance.json`. Review them before packaging any upstream "
        "bytes: this project redistributes none of them.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_status_report(layout, cfg) -> dict[str, Path]:
    """Assemble the status report from whatever artefacts exist."""
    manifest_meta = _read_json(layout.manifest_meta)
    plan = _read_json(layout.data / "collection" / "plan.json")
    plan_verify = _read_json(layout.data / "collection" / "plan_verify_upstream.json")
    pilot = _read_json(layout.data / "collection" / "pilot_keys.json")
    collection = _read_json(layout.data / "collection" / "collection_report.json")
    splits = _read_json(layout.splits_dir / "splits_report.json")
    adapters = _read_json(layout.adapters_dir / "cohort" / "adapter_report.json") or \
        _read_json(layout.adapters_dir / "pilot" / "adapter_report.json")
    evaluation = _read_json(layout.eval_dir / "evaluation.json")
    tasks = _read_json(layout.eval_dir / "tasks" / "tasks.json")
    views = {}
    for view in ("tx_receipt", "full_calls_receipt"):
        doc = _read_json(layout.features_dir / view / "feature_dictionary.json")
        if doc:
            views[view] = doc

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    is_fixture = bool(manifest_meta and manifest_meta.get("is_fixture"))
    lines: list[str] = [
        "# Benchmark status report",
        "",
        f"- generated: {now}",
        f"- data: **{'FIXTURE' if is_fixture else 'real Ethereum mainnet'}**",
        f"- credentials present: {credential_status()}",
        "",
    ]

    lines += ["## 1. Cohort", ""]
    if manifest_meta:
        lines += [
            f"- release `{manifest_meta['release']['release_id']}`, chain "
            f"{manifest_meta['chain_id']}",
            f"- rows: **{manifest_meta['rows']}**; labels "
            f"{manifest_meta['label_counts']}; attack classes "
            f"{manifest_meta['attack_class_counts']}",
            f"- cohort digest: `{manifest_meta['cohort_sha256']}`",
            f"- quarantined: {manifest_meta['quarantined_rows']} "
            f"{manifest_meta['quarantine_reasons']}",
            f"- full cohort report: `{layout.cohort_report.relative_to(layout.root)}`",
            "",
        ]
    else:
        lines += ["_not built yet: run `bench import-txbench`._", ""]

    lines += ["## 2. Collection", ""]
    if plan:
        upstream = plan.get("upstream_supplied", {})
        lines += [
            f"- transactions planned: {plan.get('transactions')}",
            f"- upstream already supplies: envelope complete "
            f"{upstream.get('envelope_complete')}, receipt {upstream.get('receipt_present')}, "
            f"complete call tree {upstream.get('call_tree_complete')}, "
            f"with nesting {upstream.get('call_tree_with_nesting')}, "
            f"trace missing {upstream.get('trace_missing')}",
            f"- calls still needed: {plan.get('planned_calls_total')} "
            f"{plan.get('planned_calls_by_method')}",
            f"- cache hits skipped: {plan.get('cache_hits_by_method')}",
            "",
            "### Cost estimate",
            "",
        ]
        estimates = (plan.get("estimates") or {}).get("providers", {})
        lines += [_table(
            [[name, e.get("unit"), e.get("calls_total"), e.get("estimated_units"),
              e.get("run_unit_budget"), e.get("plan_unit_allowance"),
              e.get("credential_present")]
             for name, e in sorted(estimates.items())],
            ["provider", "unit", "calls", "estimated units", "run budget", "plan allowance",
             "key present"]), ""]
    else:
        lines += ["_not planned yet: run `bench collect --dry-run`._", ""]
    if plan_verify:
        estimates = (plan_verify.get("estimates") or {}).get("providers", {})
        lines += [
            "### Cross-check scenario (`collect --dry-run --verify-upstream`)",
            "",
            f"Re-reading the envelope and receipt for all "
            f"{plan_verify.get('transactions')} transactions - to verify the release against "
            f"the chain rather than to fill a gap - would cost:",
            "",
            _table([[name, e.get("unit"), e.get("calls_total"), e.get("estimated_units"),
                     e.get("plan_unit_allowance"), e.get("share_of_plan_allowance")]
                    for name, e in sorted(estimates.items())],
                   ["provider", "unit", "calls", "estimated units", "plan allowance",
                    "share of allowance"]),
            "",
        ]
    if pilot:
        lines += [f"- pilot on record: {len(pilot.get('keys') or [])} transactions "
                  f"(seed {pilot.get('seed')}), selection: {pilot.get('selection')}", ""]
    if collection:
        outcome = collection.get("outcome", {})
        lines += [
            f"- calls made: {outcome.get('calls_made')}; failures: "
            f"{outcome.get('failure_count')}; stored bytes: "
            f"{outcome.get('stored_payload_bytes')}",
            f"- budget: {outcome.get('budget')}",
            f"- reconciliation checked: {outcome.get('reconciliation_checked')}; mismatches: "
            f"{len(outcome.get('reconciliation_mismatches') or [])}",
            "",
        ]

    lines += ["## 3. Feature views", ""]
    if views:
        rows = []
        for name, doc in sorted(views.items()):
            coverage = doc.get("coverage", {})
            rows.append([doc.get("view"), coverage.get("eligible_rows"),
                         coverage.get("usable_rows"), coverage.get("usable_share"),
                         (doc.get("vocabulary") or {}).get("tokens")])
        lines += [_table(rows, ["view", "eligible", "usable", "share", "vocabulary tokens"]), ""]
        for name, doc in sorted(views.items()):
            reasons = (doc.get("coverage") or {}).get("unusable_by_reason") or {}
            if reasons:
                lines += [f"**{doc.get('view')}** unusable reasons:", ""]
                lines += [_table([[k, v] for k, v in sorted(reasons.items(),
                                                            key=lambda kv: -kv[1])],
                                 ["reason", "rows"]), ""]
    else:
        lines += ["_not built yet: run `bench build-features --view tx-receipt`._", ""]

    lines += ["## 4. Detector inputs", ""]
    if adapters:
        coverage = adapters.get("coverage", {})
        lines += [_table(
            [[key, slot.get("eligible"), slot.get("usable"), slot.get("partial"),
              slot.get("blocked"),
              ", ".join(sorted((slot.get("missing_field_counts") or {}).keys())[:4]) or "-",
              ", ".join(slot.get("attainable_evidence") or [])]
             for key, slot in sorted(coverage.items())],
            ["detector", "eligible", "usable", "partial", "blocked", "missing fields",
             "evidence attainable"]), ""]
        smoke = adapters.get("blockscan_smoke_test")
        if smoke:
            lines += [
                "### BlockScan preprocessing smoke test",
                "",
                f"- entry point: `{smoke.get('entry_point')}`",
                f"- ran: **{smoke.get('ran')}**; tokens produced: {smoke.get('token_count')}",
                f"- fidelity: **{smoke.get('fidelity')}** - {smoke.get('fidelity_reason')}",
                f"- driver available: {smoke.get('driver_available')}; missing files: "
                f"{smoke.get('driver_missing_files')}",
                "",
            ]
        plan_block = adapters.get("enrichment_plan") or {}
        if plan_block:
            lines += ["### Cheapest next enrichment", ""]
            rows = []
            for detector, slot in sorted(plan_block.items()):
                for field_name, entry in sorted(slot.get("missing_profile", {}).items()):
                    rows.append([detector, field_name, entry.get("count"),
                                 (entry.get("source") or "")[:90]])
            lines += [_table(rows, ["detector", "missing field", "txs", "source that would "
                                                                       "supply it"]), ""]
    else:
        lines += ["_not audited yet: run `bench validate-adapters`._", ""]

    lines += ["## 5. Splits", ""]
    if splits:
        rows = []
        for name, audit in sorted((splits.get("audits") or {}).items()):
            rows.append([name, audit.get("clean"),
                         json.dumps(audit.get("partitions") or {}, sort_keys=True),
                         json.dumps(audit.get("group_overlap_counts") or {}, sort_keys=True)])
        lines += [_table(rows, ["split", "clean", "partitions", "group overlap"]), ""]
        validation = splits.get("validation_partition") or {}
        lines += [f"- validation partition: {validation.get('validation_rows')} rows "
                  f"({validation.get('realised_fraction')}) from "
                  f"`{validation.get('source_split')}`, method: {validation.get('method')}",
                  f"- corrected split needed: "
                  f"{(splits.get('corrected_split') or {}).get('needed')}", ""]
    else:
        lines += ["_not verified yet: run `bench verify-splits`._", ""]

    lines += ["## 6. Tasks", ""]
    if tasks:
        rows = []
        for name, task in sorted(tasks.items()):
            if not isinstance(task, dict) or "task" not in task:
                continue
            rows.append([task["task"], task.get("positives"), task.get("negatives"),
                         task.get("ambiguous_left_unresolved"),
                         ", ".join(task.get("detectors") or [])])
        lines += [_table(rows, ["task", "positives", "controls", "ambiguous (unresolved)",
                                "detectors"]), ""]
    else:
        lines += ["_not built yet: run `bench build-tasks`._", ""]

    lines += ["## 7. Evaluation", ""]
    if evaluation:
        parity = evaluation.get("upstream_evaluator_parity") or {}
        lines += [f"- upstream evaluator parity: **{parity.get('status')}**"
                  + (f" - {parity.get('reason')}" if parity.get("reason") else ""), ""]
        if evaluation.get("fixture_mode"):
            lines += ["> Fixture run: these numbers describe a prediction fixture and justify "
                      "no accuracy claim about any detector.", ""]
        rows = []
        for name, slot in sorted((evaluation.get("models") or {}).items()):
            test = slot.get("test") or {}
            rows.append([name, test.get("recall"), test.get("precision"),
                         test.get("average_precision"),
                         test.get("false_positives_per_1000_negatives"),
                         test.get("incident_weighted_recall"),
                         json.dumps(test.get("scored_by_class") or {}, sort_keys=True)])
        lines += [_table(rows, ["model", "recall", "precision", "avg precision",
                                "FP / 1k negatives", "incident-weighted recall",
                                "scored by class"]), ""]
    else:
        lines += ["_no predictions scored yet: run `bench evaluate --predictions ...`._", ""]

    lines += ["## Rebuild", "", "```sh", *REBUILD_STEPS[1:], "```", ""]

    path = layout.reports_dir / "status.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    written = {"status_report": path}

    summary = {
        "generated_utc": now,
        "is_fixture": is_fixture,
        "cohort": manifest_meta and {
            "rows": manifest_meta["rows"], "labels": manifest_meta["label_counts"],
            "attack_classes": manifest_meta["attack_class_counts"],
            "cohort_sha256": manifest_meta["cohort_sha256"],
            "release_id": manifest_meta["release"]["release_id"]},
        "collection": plan and {"planned_calls": plan.get("planned_calls_total"),
                                "estimates": plan.get("estimates")},
        "feature_views": {k: (v.get("coverage") or {}) for k, v in views.items()},
        "detector_coverage": adapters and adapters.get("coverage"),
        "splits_clean": splits and {k: v.get("clean")
                                    for k, v in (splits.get("audits") or {}).items()},
        "evaluator_parity": evaluation and
        (evaluation.get("upstream_evaluator_parity") or {}).get("status"),
        "artifacts": _artifact_index(layout),
    }
    summary_path = layout.reports_dir / "status.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
                            encoding="utf-8")
    written["status_json"] = summary_path
    return written


def _artifact_index(layout) -> dict[str, str]:
    candidates = {
        "version_manifest": layout.manifest_meta,
        "label_table": layout.manifest,
        "schema_mapping": layout.schema_mapping,
        "quarantine": layout.quarantine,
        "release_provenance": layout.manifest_dir / "release_provenance.json",
        "cohort_report": layout.cohort_report,
        "collection_inventory": layout.data / "collection" / "inventory.parquet",
        "collection_plan": layout.data / "collection" / "plan.json",
        "collection_report": layout.data / "collection" / "collection_report.json",
        "raw_cache_index": layout.data / "collection" / "cache_index.parquet",
        "cache_sqlite": layout.cache_index,
        "splits_report": layout.splits_dir / "splits_report.json",
        "tasks": layout.eval_dir / "tasks" / "tasks.json",
        "adapter_report_cohort": layout.adapters_dir / "cohort" / "adapter_report.json",
        "adapter_report_pilot": layout.adapters_dir / "pilot" / "adapter_report.json",
        "compatibility_table_cohort": layout.adapters_dir / "cohort" / "compatibility_table.parquet",
        "evaluation": layout.eval_dir / "evaluation.json",
        "features_tx_receipt": layout.features_dir / "tx_receipt" / "feature_dictionary.json",
        "features_full_calls": layout.features_dir / "full_calls_receipt" / "feature_dictionary.json",
    }
    out = {}
    for name, path in candidates.items():
        if Path(path).exists():
            try:
                out[name] = str(Path(path).relative_to(layout.root))
            except ValueError:
                out[name] = str(path)
    return out
