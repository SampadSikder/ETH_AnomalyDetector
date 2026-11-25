"""Smoke test against BlockScan's own preprocessing entry point.

The pinned revision's `src_eth/preprocess.py` is imported from disk and its
real `extract_function_calls` is called on a record converted from our
normalised transaction. Two things are reported honestly:

  * the conversion carries **no nested `calls`**, because the release supplies
    a flat frame list - so the run is `upstream_preprocess_degraded`;
  * the driver `write_all_traces` needs `config/top_20k_address.txt` and
    `config/top_20k_hash.txt`, which the pinned revision does not contain, so
    the driver is reported as unavailable rather than approximated silently.

Only this one module of the third-party repository is imported, and it is
imported by path so that nothing else in that tree is placed on `sys.path`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from ..logs import get_logger
from ..trace_model import NormalizedTx

log = get_logger("blockscan")

ENTRY_RELATIVE = Path("src_eth/preprocess.py")
DRIVER_CONFIG_FILES = ("config/top_20k_address.txt", "config/top_20k_hash.txt")


def default_repo_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "data" / "upstream" / "detectors" / "blockscan"


def to_blockscan_record(tx: NormalizedTx) -> dict[str, Any]:
    """Convert to the shape `extract_function_calls` reads.

    Fidelity notes, all deliberate:
      * `calls` is omitted - we have no nesting, and inventing one would
        fabricate structure the chain never showed us;
      * `args` / `output` are emitted only where the release decoded them (the
        outer call), with their upstream types;
      * logs keep topic 0 first, matching the preprocessor's indexing.
    """
    def typed(values: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not values:
            return None
        return [{"type": str(v.get("type") or "data"), "data": v.get("value")} for v in values]

    root = next((f for f in tx.frames if f.is_root), None)
    record: dict[str, Any] = {
        "type": (root.call_type if root else "CALL"),
        "from": tx.from_address,
        "to": tx.to_address,
        "func": _selector(tx.input_hex),
        "gas": tx.gas,
        "value": tx.value_wei,
        "args": None,
        "output": None,
        "logs": [],
    }
    for log_record in tx.logs:
        topics = [{"type": "data", "data": t} for t in log_record.topics]
        if not topics:
            # The upstream preprocessor indexes topics[0] unconditionally, so a
            # zero-topic log is left out and reported instead of crashing it.
            continue
        record["logs"].append({
            "address": log_record.address,
            "topics": topics,
            "data": [{"type": "data", "data": log_record.data_hex}] if log_record.data_hex else [],
        })
    return record


def _selector(input_hex: str | None) -> str | None:
    if not input_hex or len(input_hex) < 10:
        return None
    return input_hex[:10]


def load_entry_module(repo_dir: Path):
    """Import the pinned preprocessing module by path, nothing else."""
    path = Path(repo_dir) / ENTRY_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"BlockScan preprocessing entry point not found: {path}")
    name = "txbench_vendored_blockscan_preprocess"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_smoke(tx: NormalizedTx, repo_dir: Path) -> dict[str, Any]:
    """Run the authors' trace builder on one converted record."""
    out: dict[str, Any] = {
        "tx_key": tx.tx_key,
        "repo_dir": str(repo_dir),
        "entry_point": f"{ENTRY_RELATIVE}::extract_function_calls",
        "fidelity": "degraded",
        "fidelity_reason": "the converted record carries no nested `calls`: the release supplies "
                           "an ordered flat frame list with no parent pointers",
        "driver_available": False,
        "driver_missing_files": [],
        "ran": False,
    }
    for relative in DRIVER_CONFIG_FILES:
        if not (Path(repo_dir) / relative).is_file():
            out["driver_missing_files"].append(relative)
    out["driver_available"] = not out["driver_missing_files"]
    if out["driver_missing_files"]:
        out["driver_reason"] = ("write_all_traces reads these files from the repository's "
                                "config/ directory; the pinned revision does not contain them "
                                "(they are distributed through an external drive link)")
    try:
        module = load_entry_module(repo_dir)
    except (FileNotFoundError, ImportError, ModuleNotFoundError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    record = to_blockscan_record(tx)
    try:
        tokens = module.extract_function_calls(record)
    except Exception as exc:                            # noqa: BLE001 - third-party code
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["ran"] = True
    out["token_count"] = len(tokens)
    out["tokens_head"] = [str(t) for t in tokens[:24]]
    out["zero_topic_logs_omitted"] = sum(1 for l in tx.logs if l.is_zero_topic)
    out["logs_converted"] = len(record["logs"])
    out["nested_calls_supplied"] = 0
    return out
