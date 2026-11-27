"""BlockScan, run through the authors' own pipeline end to end.

Four things have to be in place before a single transaction can be scored, and
each is checked by name rather than assumed:

  * `config/top_20k_address.txt` and `config/top_20k_hash.txt` - the
    vocabulary allow-lists. Every address or selector outside them becomes
    `[OOV]` and is then **deleted** from the sequence (`remove_oov=True` is the
    default in the authors' `write_all_traces`), so these files decide what the
    model ever sees;
  * `config/tokenizer` - the authors' tokenizer;
  * `model/rope_roberta` - the pretrained checkpoint;
  * an interpreter with the authors' pinned dependency set (torch 2.1.0,
    transformers 4.32.0), which cannot be this project's own environment.

The pinned revision ships none of the first three: the README distributes them
through a Google Drive link. `asset_report` states exactly which are present,
and the run stops with that report rather than approximating anything.

The sequence is the authors': our normalised record -> `to_blockscan_record`
(the same conversion the adapter smoke test uses) -> their
`write_all_traces` (OOV handling, hex normalisation, `[OOV]` removal) ->
their tokenizer -> their checkpoint -> their `compute_abnormal_score`.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..adapters.blockscan_smoke import to_blockscan_record
from ..errors import TxBenchError
from ..hashutil import sha256_file
from ..logs import get_logger
from ..trace_model import NormalizedTx
from . import PredictionRow, STATUS_FAILED, STATUS_SCORED

log = get_logger("blockscan.run")

MODEL_NAME = "blockscan"
#: The checkpoint identifies the variant: the authors' rope-roberta foundation
#: model, run zero-shot (it is not fine-tuned on this project's transactions).
MODEL_VERSION = "pinned-rope-roberta-mlm"

REQUIRED_ASSETS = {
    "vocabulary_addresses": "config/top_20k_address.txt",
    "vocabulary_selectors": "config/top_20k_hash.txt",
    "tokenizer": "config/tokenizer",
    "checkpoint": "model/rope_roberta",
}
CHECKPOINT_WEIGHTS = ("model.safetensors", "pytorch_model.bin")
TOKENIZER_FILES = ("tokenizer.json", "vocab.txt", "tokenizer_config.json")

SCORER = Path(__file__).with_name("blockscan_score.py")


def asset_report(repo_dir: Path, *, checksums: bool = False) -> dict[str, Any]:
    """What is present, what is missing, and where the missing part comes from.

    With `checksums=True` every present file is hashed, so a run artifact can
    pin exactly which copy of the authors' release produced its numbers.
    """
    repo_dir = Path(repo_dir)
    present: dict[str, Any] = {}
    missing: list[dict[str, str]] = []
    for name, relative in REQUIRED_ASSETS.items():
        path = repo_dir / relative
        if name == "tokenizer":
            files = [f for f in TOKENIZER_FILES if (path / f).is_file()]
            if len(files) < 2:
                missing.append({"asset": name, "path": relative,
                                "needed_by": "transformers AutoTokenizer.from_pretrained",
                                "source": "the authors' Drive release, folder config/tokenizer"})
                continue
            present[name] = {"path": str(path), "files": files}
            if checksums:
                present[name]["sha256"] = {f: sha256_file(path / f) for f in files}
        elif name == "checkpoint":
            weights = [f for f in CHECKPOINT_WEIGHTS if (path / f).is_file()]
            if not (path / "config.json").is_file() or not weights:
                missing.append({"asset": name, "path": relative,
                                "needed_by": "transformers AutoModelForMaskedLM.from_pretrained",
                                "source": "the authors' Drive release, model/model.zip"})
                continue
            present[name] = {"path": str(path), "weights": weights,
                             "bytes": sum((path / f).stat().st_size for f in weights)}
            if checksums:
                # The file transformers actually loads is hashed first.
                present[name]["sha256"] = {f: sha256_file(path / f)
                                           for f in (*weights, "config.json")}
        else:
            if not path.is_file():
                missing.append({"asset": name, "path": relative,
                                "needed_by": "write_all_traces: tokens outside these lists "
                                             "become [OOV] and are then removed",
                                "source": "the authors' Drive release, folder config/"})
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            present[name] = {"path": str(path), "entries": len(lines),
                             "bytes": path.stat().st_size}
            if checksums:
                present[name]["sha256"] = sha256_file(path)
    return {"repo_dir": str(repo_dir), "present": present, "missing": missing,
            "complete": not missing}


def resolve_scorer_python(explicit: str | None = None) -> Path:
    """The interpreter holding the authors' pinned dependency set."""
    candidates = [explicit, os.environ.get("BLOCKSCAN_PYTHON")]
    base = Path(sys.prefix).parent.parent if (Path(sys.prefix).parent.name == "envs") else None
    if base is not None:
        candidates.append(str(base / "envs" / "blockscan" / "bin" / "python"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise TxBenchError(
        "no interpreter with BlockScan's pinned dependencies (torch 2.1.0, transformers "
        "4.32.0). Point --scorer-python or BLOCKSCAN_PYTHON at one; this project's own "
        "environment cannot hold that set, because it needs numpy 2 and transformers 4.32 "
        "needs numpy 1.")


@dataclass
class StagedPartition:
    partition: str
    directory: Path
    keys: list[str]
    raw_token_counts: dict[str, int]


def stage_records(records: Sequence[tuple[str, NormalizedTx | None]],
                  partitions: dict[str, str], out_dir: Path, *,
                  extract_function_calls) -> tuple[dict[str, StagedPartition], list[str]]:
    """Write one `{tx_hash: [record]}` file per transaction, grouped by partition.

    The file name is the transaction hash, which is how `tx_path.txt` maps back
    to a key afterwards. A record the conversion cannot produce is returned as
    a failure rather than written as an empty trace.
    """
    out_dir = Path(out_dir)
    staged: dict[str, StagedPartition] = {}
    failures: list[str] = []
    for tx_key, tx in records:
        partition = partitions.get(tx_key)
        if partition is None or tx is None:
            failures.append(tx_key)
            continue
        slot = staged.get(partition)
        if slot is None:
            directory = out_dir / partition
            directory.mkdir(parents=True, exist_ok=True)
            slot = staged[partition] = StagedPartition(partition=partition, directory=directory,
                                                       keys=[], raw_token_counts={})
        record = to_blockscan_record(tx)
        tx_hash = tx_key.split(":", 1)[1]
        (slot.directory / f"{tx_hash}.json").write_text(
            json.dumps({tx_hash: [record]}), encoding="utf-8")
        slot.keys.append(tx_key)
        slot.raw_token_counts[tx_key] = len(extract_function_calls(record))
    return staged, failures


def run_pinned_preprocess(preprocess_module, *, input_dir: Path, config_dir: Path,
                          output_dir: Path) -> None:
    """The authors' `write_all_traces`, called as written."""
    preprocess_module.write_all_traces(str(input_dir), str(config_dir), str(output_dir))


def read_preprocessed(directory: Path) -> tuple[list[str], list[str]]:
    lines = (Path(directory) / "preprocessed_tx.txt").read_text(encoding="utf-8").splitlines()
    paths = (Path(directory) / "tx_path.txt").read_text(encoding="utf-8").splitlines()
    return lines, paths


def key_of_path(path: str, chain_id: int) -> str:
    return f"{chain_id}:{Path(path.strip()).stem.lower()}"


def score_partition(scorer_python: Path, *, repo_dir: Path, data_dir: Path, out_json: Path,
                    seq_len: int = 1024, mask_percent: int = 15, top_s: int = 3,
                    seed: int = 42, timeout_s: int = 36000,
                    decode_out: Path | None = None) -> dict[str, Any]:
    """Run the pinned model in its own interpreter and read back the scores."""
    command = [str(scorer_python), str(SCORER), "--repo", str(repo_dir),
               "--data-dir", str(data_dir), "--out", str(out_json),
               "--seq-len", str(seq_len), "--mask-percent", str(mask_percent),
               "--top-s", str(top_s), "--numpy-seed", str(seed)]
    if decode_out is not None:
        command += ["--decode-out", str(decode_out)]
    log.info("scoring %s with the pinned model", data_dir)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s,
                               check=False)
    if completed.returncode != 0 or not Path(out_json).is_file():
        raise TxBenchError(
            f"the pinned BlockScan scorer failed (exit {completed.returncode}). "
            f"stderr tail: {completed.stderr[-1500:]}")
    payload = json.loads(Path(out_json).read_text(encoding="utf-8"))
    payload["command"] = command
    return payload


def rows_from_scores(payload: dict[str, Any], *, chain_id: int, partition: str,
                     raw_token_counts: dict[str, int]) -> list[PredictionRow]:
    """Turn the scorer's output into prediction rows, failures included."""
    rows: list[PredictionRow] = []
    for record in payload["rows"]:
        tx_key = key_of_path(record["path"], chain_id)
        raw = raw_token_counts.get(tx_key)
        kept = record["trace_token_words"]
        extra = {
            "tokens_before_truncation": record["tokens_before_truncation"],
            "tokens_scored": record["tokens_scored"],
            "masked_tokens": record["masked_tokens"],
            "truncated": record["truncated"],
            "trace_tokens_after_oov_removal": kept,
            "trace_tokens_before_oov_removal": raw,
            "oov_tokens_removed": None if raw is None else raw - kept,
            # The pinned score is a count of wrongly predicted masked tokens, so it grows
            # with sequence length. The rate is kept beside it for anyone who needs a
            # length-independent view; the submitted score stays the authors' own.
            "abnormal_rate": (None if record["score"] is None or not record["masked_tokens"]
                              else record["score"] / record["masked_tokens"]),
        }
        if record["score"] is None:
            rows.append(PredictionRow(
                tx_key=tx_key, model=MODEL_NAME, version=MODEL_VERSION, split=partition,
                status=STATUS_FAILED,
                note="the pinned scorer returned no score for this trace", extra=extra))
            continue
        rows.append(PredictionRow(
            tx_key=tx_key, model=MODEL_NAME, version=MODEL_VERSION, split=partition,
            status=STATUS_SCORED, score=float(record["score"]),
            score_direction="higher_is_attack",
            note=("truncated to the authors' 1024-token limit" if record["truncated"] else None),
            extra=extra))
    return rows


def truncation_summary(rows: Iterable[PredictionRow]) -> dict[str, Any]:
    """What the authors' 1024-token limit and OOV removal did to our traces."""
    rows = list(rows)
    before = [r.extra["tokens_before_truncation"] for r in rows
              if r.extra.get("tokens_before_truncation") is not None]
    removed = [r.extra["oov_tokens_removed"] for r in rows
               if r.extra.get("oov_tokens_removed") is not None]
    raw = [r.extra["trace_tokens_before_oov_removal"] for r in rows
           if r.extra.get("trace_tokens_before_oov_removal") is not None]
    truncated = [r for r in rows if r.extra.get("truncated")]

    def quantile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return float(ordered[position])

    return {
        "transactions": len(rows),
        "truncated_transactions": len(truncated),
        "truncated_share": round(len(truncated) / len(rows), 6) if rows else None,
        "model_tokens_before_truncation": {
            "min": min(before, default=None), "median": quantile(before, 0.5),
            "p90": quantile(before, 0.9), "max": max(before, default=None)},
        "trace_tokens_before_oov_removal": {
            "min": min(raw, default=None), "median": quantile(raw, 0.5),
            "max": max(raw, default=None), "total": sum(raw)},
        "oov_tokens_removed": {"total": sum(removed), "median": quantile(removed, 0.5),
                               "max": max(removed, default=None)},
        "oov_removal_share_of_trace": (round(sum(removed) / sum(raw), 6)
                                       if raw and sum(raw) else None),
        "notes": [
            "the authors' tokenizer call passes truncation=True, max_length=1024, so a "
            "longer trace is cut and its tail never reaches the model",
            "write_all_traces replaces every address or selector outside the two allow-lists "
            "with [OOV] and then deletes it (remove_oov=True, the authors' default), so the "
            "share above is sequence the model never sees",
            "the abnormal score counts wrongly predicted masked tokens, and the number of "
            "masked tokens is 15% of the padded length, so the score's scale depends on how "
            "long a trace survived truncation",
        ],
    }


def clear_directory(path: Path) -> None:
    if Path(path).exists():
        shutil.rmtree(path)


# --- the smoke test ----------------------------------------------------------

def load_allow_list(repo_dir: Path) -> set[str]:
    """The addresses and selectors `write_all_traces` keeps; everything else goes."""
    keep: set[str] = set()
    for relative in ("config/top_20k_address.txt", "config/top_20k_hash.txt"):
        path = Path(repo_dir) / relative
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            token = line.split(":")[0].strip().lower()
            if token:
                keep.add(token)
    return keep


def choose_smoke_subject(records: Sequence[tuple[str, NormalizedTx | None]],
                         partitions: dict[str, str], allow_list: set[str], *,
                         partition: str = "train") -> dict[str, Any]:
    """One transaction whose nesting can be *proved* to reach the model input.

    Preference order, all of it stated rather than left to luck: the record must
    carry real nesting; the child frame used as evidence must survive the OOV
    allow-list, or nothing distinctive about it would be left in the sequence;
    and a shorter trace is preferred, because the authors' tokenizer truncates
    at 1024 tokens and evidence beyond the cut would not reach the model.
    """
    candidates = []
    for tx_key, tx in records:
        if tx is None or partitions.get(tx_key) != partition:
            continue
        if not tx.call_tree_has_nesting:
            continue
        children = [f for f in tx.frames
                    if not f.is_root and f.parent_index is not None and f.to_address]
        if not children:
            continue
        allowed = [f for f in children if (f.to_address or "").lower() in allow_list]
        candidates.append({"tx_key": tx_key, "frames": len(tx.frames),
                           "children": len(children),
                           "child": (allowed or children)[0],
                           "child_in_allow_list": bool(allowed)})
    if not candidates:
        raise TxBenchError(
            f"no {partition} transaction carries a recovered nested call with a callee "
            "address, so the smoke test has nothing to prove")
    candidates.sort(key=lambda c: (not c["child_in_allow_list"], c["frames"]))
    return candidates[0]


def nested_call_evidence(subject: dict[str, Any], decoded_text: str, *,
                         logs: int) -> dict[str, Any]:
    """Did a nested call reach the model input? Answered on the decoded ids.

    The text here is what the scorer fed the model, decoded back, so this is not
    an argument about the conversion: either the child frame is in the sequence
    the model saw or it is not.
    """
    child = subject["child"]
    needle = (child.to_address or "").lower()
    blocks = decoded_text.count("[START]")
    # the root call and each log occupy one block each; anything beyond them is a
    # nested call that survived OOV removal and truncation
    nested_blocks = max(0, blocks - 1 - logs)
    return {
        "subject": subject["tx_key"],
        "frames_in_record": subject["frames"],
        "child_frames_in_record": subject["children"],
        "evidence_child": {"frame_index": child.index, "call_type": child.call_type,
                           "to_address": child.to_address,
                           "parent_index": child.parent_index, "depth": child.depth,
                           "in_oov_allow_list": subject["child_in_allow_list"]},
        "needle": needle,
        "needle_in_model_input": needle in decoded_text.lower() if needle else None,
        "blocks_in_model_input": blocks,
        "logs_in_record": logs,
        "nested_call_blocks_in_model_input": nested_blocks,
        "nested_call_reached_model": nested_blocks > 0,
    }
