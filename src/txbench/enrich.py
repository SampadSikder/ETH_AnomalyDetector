"""The call-trace backfill: pick the blocked transactions, recover their trees,
rebuild the detector inputs that the missing nesting was blocking.

The pilot is defined by the coverage output, not by a hand-written list: the
hashes are whatever `bench validate-adapters` last recorded as blocked for
DeFiGuard. That keeps the job honest when the coverage changes, and it is why
the selector reports its own count instead of asserting 42.

Caching is the same content-addressed store the RPC collector uses, keyed by
chain, hash, method and a neutral tracer-config string, so a completed pilot
re-runs with no query at all.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from . import bq
from .cache import CacheRef, CacheStore
from .calltree import (TRACE_SOURCE_DEFAULT, TraceRow, apply_nesting, cross_check_root,
                       parse_trace_path, reconstruct)
from .errors import TxBenchError
from .flows import build_flow_graph, defiguard_node_features
from .hashutil import canonical_json
from .logs import get_logger

log = get_logger("enrich")

#: Cache method name for a warehouse trace read. Neutral: no class information.
CACHE_METHOD = "bigquery_traces"

COVERAGE_COLUMNS = (
    "tx_hash", "tx_key", "label", "split_temporal", "block_number", "block_date",
    "frame_count", "tree_complete", "reverted_ancestry_resolved",
    "defiguard_graph_rebuilt", "blockscan_preprocess_ok", "missing_reason",
)


def tracer_config_for(table: str) -> str:
    """A short, neutral cache discriminator derived from the source table."""
    return table.split(".")[-2] + "_" + table.split(".")[-1] if "." in table else table


# --- 1. pilot selection -------------------------------------------------------

@dataclass
class Pilot:
    frame: pd.DataFrame
    source: Path
    expected: int | None = None

    @property
    def hashes(self) -> list[str]:
        return [h.lower() for h in self.frame["tx_hash"].tolist()]

    @property
    def dates(self) -> list[str]:
        return sorted(set(self.frame["block_date"].tolist()))

    def blocks_by_date(self, hashes: Iterable[str] | None = None
                       ) -> dict[str, list[int]]:
        """Exact block numbers per UTC day, which narrow the scan sharply.

        Each hash sits in one block, and the frozen manifest records which, so
        this is an exact filter rather than a heuristic bound.
        """
        wanted = {h.lower() for h in hashes} if hashes is not None else None
        out: dict[str, set[int]] = {}
        for row in self.frame.itertuples(index=False):
            if wanted is not None and str(row.tx_hash).lower() not in wanted:
                continue
            out.setdefault(str(row.block_date), set()).add(int(row.block_number))
        return {day: sorted(blocks) for day, blocks in sorted(out.items())}

    def discrepancy(self) -> str | None:
        if self.expected is None or len(self.frame) == self.expected:
            return None
        return (f"the coverage output lists {len(self.frame)} blocked transaction(s), not the "
                f"{self.expected} expected. The coverage table is authoritative; re-run "
                "`bench validate-adapters` if it is stale.")

    def summary(self) -> dict[str, Any]:
        by_reason = self.frame["blocked_reason_kind"].value_counts().to_dict()
        return {
            "source": str(self.source),
            "hashes": len(self.frame),
            "unique_hashes": int(self.frame["tx_hash"].nunique()),
            "distinct_partition_days": len(self.dates),
            "block_range": [int(self.frame.block_number.min()),
                            int(self.frame.block_number.max())],
            "date_range": [min(self.dates), max(self.dates)],
            "labels": {str(k): int(v) for k, v in
                       self.frame["label"].value_counts().to_dict().items()},
            "split_temporal": self.frame["split_temporal"].value_counts().to_dict(),
            "blocked_reason_kinds": {str(k): int(v) for k, v in by_reason.items()},
            "recoverable_by_call_tree": int((self.frame["blocked_reason_kind"]
                                             == "failed_frames").sum()),
        }


def select_pilot(layout, manifest: pd.DataFrame, *, detector: str = "defiguard",
                 expected: int | None = None, scope: str = "cohort",
                 select: str = "blocked") -> Pilot:
    """Hashes from the coverage output, joined to the frozen manifest.

    `select="blocked"` is the pilot: only the transactions a detector cannot
    use. `select="all"` takes every transaction in the frozen set, which is
    one coverage row per transaction whatever its verdict.
    """
    if select not in ("blocked", "all"):
        raise TxBenchError(f"unknown selection {select!r}: expected 'blocked' or 'all'")
    table_path = layout.adapters_dir / scope / "compatibility_table.parquet"
    if not table_path.is_file():
        raise TxBenchError(
            f"no coverage output at {table_path}. Run `bench validate-adapters` first.")
    coverage = pd.read_parquet(table_path)
    chosen = coverage[coverage["detector"] == detector]
    if select == "blocked":
        chosen = chosen[chosen["verdict"] == "blocked"]
    blocked = chosen.copy()
    if blocked.empty:
        raise TxBenchError(
            f"{detector} has no {'blocked ' if select == 'blocked' else ''}"
            f"transactions in {table_path}")
    if blocked["tx_key"].duplicated().any():
        raise TxBenchError(
            f"{table_path} holds duplicate tx_key rows for {detector}; the frozen set "
            "must carry exactly one row per transaction")

    def reason_kind(raw: Any) -> str:
        try:
            reasons = json.loads(raw) if isinstance(raw, str) else dict(raw or {})
        except (TypeError, ValueError):
            return "unparsed"
        native = str(reasons.get("native_flows", ""))
        if "pre_byzantium" in native:
            return "pre_byzantium_receipt"
        if "frame(s) failed" in native:
            return "failed_frames"
        return "other"

    blocked["blocked_reason_kind"] = blocked["missing_field_reason"].map(reason_kind)
    blocked["blocked_reason"] = blocked["missing_field_reason"]

    wanted = ["tx_key", "tx_hash", "label", "attack_class", "block_number", "block_timestamp",
              "split_temporal", "group_incident"]
    joined = blocked.merge(manifest[wanted], on="tx_key", how="left")
    unmatched = joined[joined["tx_hash"].isna()]
    if not unmatched.empty:
        raise TxBenchError(f"{len(unmatched)} blocked tx_key(s) are absent from the manifest; "
                           "the coverage output and the manifest disagree")
    stamps = pd.to_datetime(joined["block_timestamp"], unit="s", errors="coerce", utc=True)
    joined["block_date"] = stamps.dt.strftime("%Y-%m-%d")
    if joined["block_date"].isna().any():
        raise TxBenchError("some blocked transactions have no usable block timestamp, so their "
                           "partition cannot be determined")
    joined = joined.sort_values("tx_key").reset_index(drop=True)
    return Pilot(frame=joined, source=table_path, expected=expected)


# --- 2. rows -> traces --------------------------------------------------------

def jsonable(value: Any) -> Any:
    """A warehouse value as JSON, losing nothing that matters.

    BigQuery hands back `datetime`, `Decimal` and `bytes` objects, none of
    which `json.dumps` accepts. Converting them here, at the fetch boundary,
    is what makes a freshly queried row and the same row read back out of the
    cache identical: a round trip through the cache must not change what the
    reconstruction sees, or a re-run would not be a re-run.

    A `Decimal` becomes a digit string rather than a float, because wei does
    not survive 53 bits of mantissa.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return str(int(value))
        return format(value, "f")
    if isinstance(value, (bytes, bytearray)):
        return "0x" + bytes(value).hex()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return str(value)


def jsonable_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every row's every value, JSON-safe and cache-safe."""
    return [{str(key): jsonable(item) for key, item in row.items()} for row in rows]


def _to_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text, 16) if text.lower().startswith("0x") else int(text)
    except ValueError:
        return None


def _error_of(row: dict[str, Any]) -> str | None:
    """A frame's failure, from whichever of the two conventions the table uses."""
    error = row.get("error")
    if error not in (None, "", "null"):
        return str(error)
    status = row.get("status")
    if status is None:
        return None
    number = _to_int(status)
    if number is None:
        # A boolean-ish string.
        if str(status).strip().lower() in ("false", "f", "no"):
            return "status=false"
        return None
    return None if number == 1 else "status=0"


def rows_to_traces(rows: Iterable[dict[str, Any]]) -> dict[str, list[TraceRow]]:
    """Group warehouse rows by transaction and turn each into a `TraceRow`.

    The rows arrive aliased to role names by `bq.build_query`, so this function
    reads roles, not the table's own column names.
    """
    out: dict[str, list[TraceRow]] = {}
    for row in rows:
        tx_hash = str(row.get("tx_hash") or "").lower()
        if not tx_hash:
            continue
        out.setdefault(tx_hash, []).append(TraceRow(
            path=parse_trace_path(row.get("call_path")),
            # `action.call_type` is null for a create, a suicide or a reward,
            # where the frame's kind lives in `trace_type` instead.
            call_type=row.get("call_type") or row.get("trace_type"),
            from_address=row.get("from_address"),
            to_address=row.get("to_address"),
            value_wei=_to_int(row.get("value")),
            input_hex=row.get("input"),
            output_hex=row.get("output"),
            error=_error_of(row),
            gas=_to_int(row.get("gas")),
            gas_used=_to_int(row.get("gas_used")),
            subcall_count=_to_int(row.get("subcall_count")),
            source=TRACE_SOURCE_DEFAULT,
        ))
    return out


# --- applying recovered trees to the records the adapters receive -------------

#: Fields the warehouse query does not carry but the release does, so they are
#: taken back from the release's own frames rather than lost. A CREATE's bytes
#: live in `action.init` and a SELFDESTRUCT has no calldata at all, so the
#: trace rows leave these empty where the release does not.
CARRIED_FIELDS = ("input_hex", "output_hex", "gas", "gas_used")


def carry_release_payloads(previous: list[Any], tree) -> dict[str, Any]:
    """Put back the per-frame bytes the release supplied and the query did not.

    Applying a tree replaces the frame list wholesale, which would throw away
    real data: the release carries init code for every CREATE and a payload for
    every SELFDESTRUCT, while `action.input` is empty for both.

    The two lists are matched by position, in depth-first pre-order - which for
    a path tuple is plain lexicographic order. That alignment is a hypothesis,
    so it is **verified** on `from_address`, `to_address` and `value_wei` for
    every frame before a single byte is carried, and abandoned whole if any
    frame disagrees or the counts differ. Order is never allowed to decide
    nesting; it only lines up two descriptions of frames whose identity is
    then checked independently.
    """
    out: dict[str, Any] = {"carried": 0, "frames_matched": 0, "reason": None}
    if not previous or not tree.frames:
        out["reason"] = "no frames on one side"
        return out
    if len(previous) != len(tree.frames):
        # The two descriptions list different numbers of frames, so position
        # proves nothing. Fall back to matching on a frame's own identity, and
        # only where that identity is unique on both sides - an ambiguous key
        # carries nothing.
        out["reason"] = (f"frame counts differ: release {len(previous)}, "
                         f"reconstruction {len(tree.frames)}")
        out.update(_carry_by_identity(previous, tree.frames, out))
        return out
    ordered = [tree.frames[i] for _, i in sorted(tree.index_by_path.items())]
    for old, new in zip(previous, ordered):
        for name in ("from_address", "to_address", "value_wei"):
            mine, theirs = getattr(old, name, None), getattr(new, name, None)
            if mine is not None and theirs is not None and mine != theirs:
                out["reason"] = f"frames disagree on {name}, so they are not the same frames"
                return out
    for old, new in zip(previous, ordered):
        out["frames_matched"] += 1
        for name in CARRIED_FIELDS:
            if getattr(new, name, None) is None and getattr(old, name, None) is not None:
                setattr(new, name, getattr(old, name))
                out["carried"] += 1
    return out


def _frame_key(frame) -> tuple[Any, ...]:
    return (getattr(frame, "from_address", None), getattr(frame, "to_address", None),
            getattr(frame, "value_wei", None), getattr(frame, "subcall_count", None))


def _carry_by_identity(previous: list[Any], recovered: list[Any],
                       out: dict[str, Any]) -> dict[str, Any]:
    """Carry payloads between frames whose identity is unique on both sides."""
    def unique(frames):
        seen: dict[tuple[Any, ...], Any] = {}
        duplicated = set()
        for frame in frames:
            key = _frame_key(frame)
            if key in seen:
                duplicated.add(key)
            seen[key] = frame
        return {k: v for k, v in seen.items() if k not in duplicated}

    mine, theirs = unique(previous), unique(recovered)
    carried = out.get("carried", 0)
    matched = out.get("frames_matched", 0)
    for key, new in theirs.items():
        old = mine.get(key)
        if old is None:
            continue
        matched += 1
        for name in CARRIED_FIELDS:
            if getattr(new, name, None) is None and getattr(old, name, None) is not None:
                setattr(new, name, getattr(old, name))
                carried += 1
    return {"carried": carried, "frames_matched": matched, "matched_by": "identity"}


@dataclass
class TreeApplication:
    """What applying the recovered trees did, for the audit's provenance."""
    table: str = ""
    considered: int = 0
    applied: int = 0
    complete: int = 0
    with_nesting: int = 0
    ancestry_resolved: int = 0
    payload_frames_restored: int = 0
    payload_records_restored: int = 0
    payload_not_carried: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)
    root_disagreements: list[str] = field(default_factory=list)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_table": self.table,
            "records_considered": self.considered,
            "trees_applied": self.applied,
            "trees_complete": self.complete,
            "records_with_nesting": self.with_nesting,
            "ancestry_resolved": self.ancestry_resolved,
            "release_payload_frames_restored": self.payload_frames_restored,
            "release_payload_records_restored": self.payload_records_restored,
            "release_payload_not_carried": dict(sorted(self.payload_not_carried.items())),
            "skipped": dict(sorted(self.skipped.items())),
            "root_disagreements": sorted(self.root_disagreements),
        }


def apply_cached_trees(assembled: Iterable[tuple[str, Any]], *, store: CacheStore | None,
                       chain_id: int, table: str) -> TreeApplication:
    """Give each assembled record the call tree already recovered for it.

    This is the same reconstruction that produced the collection report -
    `reconstruct`, then `cross_check_root`, then `apply_nesting` - applied to
    the records the adapters actually receive. Coverage then reflects what a
    detector would really be handed, instead of what the release alone
    supplies.

    Nothing here contacts BigQuery or an RPC endpoint. A hash with no cached
    rows is left exactly as the release supplied it, and a tree whose root
    disagrees with the local transaction is **not** applied: a record that
    describes a different transaction is worse than a flat one.
    """
    out = TreeApplication(table=table)
    if store is None:
        return out
    for _, tx in assembled:
        if tx is None:
            continue
        out.considered += 1
        rows = cached_rows(store, chain_id, tx.tx_hash, table)
        if rows is None:
            out.skip("no cached warehouse rows for this hash")
            continue
        if not rows:
            out.skip("cached result holds no rows for this hash")
            continue
        traces = rows_to_traces(rows).get(tx.tx_hash.lower(), [])
        if not traces:
            out.skip("cached rows carry no frames for this hash")
            continue
        tree = reconstruct(tx.tx_hash, traces, source=table)
        if not tree.frames:
            out.skip("reconstruction placed no frames: " + ", ".join(
                sorted({i["issue"] for i in tree.issues})))
            continue
        check = cross_check_root(tree, tx)
        if check.get("mismatches"):
            out.root_disagreements.append(tx.tx_key)
            out.skip("root cross-check disagrees on " + ", ".join(sorted(check["mismatches"])))
            continue
        previous = list(tx.frames)
        apply_nesting(tx, tree, source=f"bigquery:{table}")
        carried = carry_release_payloads(previous, tree)
        if carried["carried"]:
            out.payload_frames_restored += carried["carried"]
            out.payload_records_restored += 1
        if carried["reason"]:
            key = carried["reason"].split(":")[0]
            out.payload_not_carried[key] = out.payload_not_carried.get(key, 0) + 1
        out.applied += 1
        out.complete += int(tree.complete)
        out.with_nesting += int(bool(tx.call_tree_has_nesting))
        out.ancestry_resolved += int(bool(tx.call_tree_ancestry_known)
                                     and all(f.reverted is not None for f in tx.frames))
    return out


def hashes_missing_creation_input(assembled: Iterable[tuple[str, Any]]) -> list[str]:
    """Transactions whose assembled record still lacks creation input bytes.

    The cache is not the measure here: for most transactions the release's own
    frames supply a CREATE's init code and the carry-over puts it back, so the
    cached row stays null while the record a detector receives is complete.
    Only what is still missing *after* assembly is worth fetching.
    """
    out: list[str] = []
    for _, tx in assembled:
        if tx is None:
            continue
        if any((f.call_type or "").upper() in ("CREATE", "CREATE2")
               and f.input_hex is None for f in tx.frames):
            out.append(tx.tx_hash.lower())
    return sorted(set(out))


def _is_creation(row: dict[str, Any]) -> bool:
    return (str(row.get("trace_type") or "").lower() in ("create", "create2")
            or str(row.get("call_type") or "").upper() in ("CREATE", "CREATE2"))


def select_init_gaps(store: CacheStore, chain_id: int, table: str,
                     hashes: Iterable[str]) -> dict[str, list[list[int]]]:
    """Cached creation frames that carry no input bytes, by transaction hash.

    Read from the cached rows, because those are what hold the trace path a
    patch has to match on - a `CallFrame` keeps its parent pointer, not its
    path. Root creation frames are excluded: a creation transaction's own
    input *is* that frame's init code, and the release carry-over supplies it
    without a query.
    """
    out: dict[str, list[list[int]]] = {}
    for tx_hash in hashes:
        for row in cached_rows(store, chain_id, tx_hash, table) or []:
            if not _is_creation(row) or row.get("input") not in (None, ""):
                continue
            path = list(parse_trace_path(row.get("call_path")))
            if not path:
                continue
            out.setdefault(tx_hash.lower(), []).append(path)
    return {h: sorted(p) for h, p in sorted(out.items())}


def patch_cached_init(store: CacheStore, chain_id: int, table: str,
                      rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Write fetched init bytes into the rows already cached for each hash.

    The cache is patched rather than replaced, so the one targeted column joins
    what the full collection already paid for. A frame whose path is not in the
    cached rows is reported, never appended: a row the reconstruction never saw
    has no verified place in the tree.
    """
    wanted: dict[str, dict[tuple[int, ...], str]] = {}
    for row in rows:
        tx_hash = str(row.get("tx_hash") or "").lower()
        init = row.get("init")
        if not tx_hash or init in (None, ""):
            continue
        path = tuple(parse_trace_path(row.get("call_path")))
        wanted.setdefault(tx_hash, {})[path] = str(init)

    out = {"hashes_patched": 0, "frames_patched": 0, "frames_unmatched": 0,
           "hashes_without_cache": 0, "bytes_written": 0}
    for tx_hash, by_path in wanted.items():
        cached = cached_rows(store, chain_id, tx_hash, table)
        if cached is None:
            out["hashes_without_cache"] += 1
            continue
        touched = 0
        index = {tuple(parse_trace_path(r.get("call_path"))): r for r in cached}
        for path, init in by_path.items():
            target = index.get(path)
            if target is None:
                out["frames_unmatched"] += 1
                continue
            if target.get("input") in (None, ""):
                target["input"] = init
                touched += 1
                out["bytes_written"] += max(0, (len(init) - 2) // 2)
        if touched:
            store_rows(store, chain_id, tx_hash, table, cached)
            out["hashes_patched"] += 1
            out["frames_patched"] += touched
    return out


# --- 3. caching ---------------------------------------------------------------

def cache_ref(chain_id: int, tx_hash: str, table: str) -> CacheRef:
    return CacheRef.for_tx(chain_id, tx_hash, CACHE_METHOD, tracer_config_for(table))


def cached_rows(store: CacheStore | None, chain_id: int, tx_hash: str, table: str
                ) -> list[dict[str, Any]] | None:
    if store is None:
        return None
    payload = store.load(cache_ref(chain_id, tx_hash, table))
    if payload is None:
        return None
    rows = payload.get("rows") if isinstance(payload, dict) else payload
    return list(rows or [])


def store_rows(store: CacheStore, chain_id: int, tx_hash: str, table: str,
               rows: list[dict[str, Any]], *, block_number: int | None = None) -> None:
    safe = jsonable_rows(rows)
    payload = {"table": table, "tx_hash": tx_hash, "rows": safe, "row_count": len(safe)}
    store.put(cache_ref(chain_id, tx_hash, table), payload, provider="bigquery",
              completeness="complete" if rows else "empty", block_number=block_number)


def partition_cached(pilot: Pilot, store: CacheStore | None, chain_id: int, table: str
                     ) -> tuple[list[str], list[str]]:
    """Split the pilot into hashes already cached and hashes still to fetch."""
    hits, misses = [], []
    for tx_hash in pilot.hashes:
        if cached_rows(store, chain_id, tx_hash, table) is not None:
            hits.append(tx_hash)
        else:
            misses.append(tx_hash)
    return hits, misses


# --- 4. rebuild ---------------------------------------------------------------

@dataclass
class Rebuild:
    tx_hash: str
    tx_key: str
    tree: Any = None
    root_check: dict[str, Any] = field(default_factory=dict)
    graph_before: dict[str, Any] = field(default_factory=dict)
    graph_after: dict[str, Any] = field(default_factory=dict)
    node_features: dict[str, Any] = field(default_factory=dict)
    blockscan: dict[str, Any] | None = None
    missing_reason: str | None = None

    @property
    def frame_count(self) -> int:
        return self.tree.frame_count if self.tree is not None else 0

    @property
    def tree_complete(self) -> bool:
        return bool(self.tree is not None and self.tree.complete)

    @property
    def ancestry_resolved(self) -> bool:
        return bool(self.graph_after.get("ancestry_resolved"))

    @property
    def graph_rebuilt(self) -> bool:
        """The native side is complete, so the graph is no longer incomplete."""
        return bool(self.graph_after.get("native_complete"))

    @property
    def blockscan_ok(self) -> bool | None:
        if self.blockscan is None:
            return None
        return bool(self.blockscan.get("ok"))

    def coverage_row(self, meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "tx_hash": self.tx_hash,
            "tx_key": self.tx_key,
            "label": meta.get("label"),
            "split_temporal": meta.get("split_temporal"),
            "block_number": meta.get("block_number"),
            "block_date": meta.get("block_date"),
            "frame_count": self.frame_count,
            "tree_complete": self.tree_complete,
            "reverted_ancestry_resolved": self.ancestry_resolved,
            "defiguard_graph_rebuilt": self.graph_rebuilt,
            "blockscan_preprocess_ok": self.blockscan_ok,
            "missing_reason": self.missing_reason,
        }


def rebuild_one(tx, rows: list[dict[str, Any]] | None, *, tx_key: str, tx_hash: str,
                table: str, blockscan_repo: Path | None = None) -> Rebuild:
    """Reconstruct one tree and rebuild the detector inputs it was blocking."""
    out = Rebuild(tx_hash=tx_hash, tx_key=tx_key)
    if tx is None:
        out.missing_reason = "no local record: the release supplies no trace for this hash"
        return out
    out.graph_before = build_flow_graph(tx).summary()

    if rows is None:
        out.missing_reason = "not fetched: no cached warehouse rows for this hash"
        return out
    if not rows:
        out.missing_reason = "no BigQuery match: the table returned no rows for this hash"
        return out

    traces = rows_to_traces(rows).get(tx_hash.lower(), [])
    if not traces:
        out.missing_reason = "no BigQuery match: rows returned but none for this hash"
        return out

    tree = reconstruct(tx_hash, traces, source=table)
    out.tree = tree
    out.root_check = cross_check_root(tree, tx)
    if not tree.frames:
        out.missing_reason = "reconstruction placed no frames: " + ", ".join(
            sorted({i["issue"] for i in tree.issues}))
        return out

    apply_nesting(tx, tree, source=f"bigquery:{table}")
    graph = build_flow_graph(tx)
    out.graph_after = graph.summary()
    out.node_features = defiguard_node_features(graph)

    if not out.graph_rebuilt:
        out.missing_reason = graph.native_reason
    elif not tree.complete:
        out.missing_reason = "tree incomplete: " + ", ".join(
            sorted({i["issue"] for i in tree.issues}))
    if out.root_check.get("mismatches"):
        fields = ", ".join(sorted(out.root_check["mismatches"]))
        out.missing_reason = ((out.missing_reason + "; ") if out.missing_reason else "") + \
            f"root cross-check disagrees on {fields}"

    if blockscan_repo is not None and blockscan_repo.is_dir():
        from .adapters.blockscan_smoke import run_smoke
        try:
            smoke = run_smoke(tx, blockscan_repo)
            out.blockscan = {"ok": smoke.get("ran", False) and not smoke.get("error"),
                             **{k: v for k, v in smoke.items() if k != "tokens_head"}}
        except Exception as exc:                       # the upstream code is not ours
            out.blockscan = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        out.blockscan = None
    return out


# --- 5. report ----------------------------------------------------------------

def write_enrichment_report(pilot: Pilot, rebuilds: Sequence[Rebuild], out_dir: Path, *,
                            extras: dict[str, Any]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_by_hash = {row.tx_hash.lower(): {
        "label": int(row.label), "split_temporal": row.split_temporal,
        "block_number": int(row.block_number), "block_date": row.block_date,
    } for row in pilot.frame.itertuples(index=False)}

    rows = [r.coverage_row(meta_by_hash.get(r.tx_hash.lower(), {})) for r in rebuilds]
    frame = pd.DataFrame(rows, columns=list(COVERAGE_COLUMNS))
    coverage_path = out_dir / "trace_backfill_coverage.parquet"
    tmp = coverage_path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(coverage_path)

    requested = len(pilot.frame)
    matched = sum(1 for r in rebuilds if r.frame_count > 0)
    recovered = sum(1 for r in rebuilds if r.tree_complete)
    unblocked = sum(1 for r in rebuilds if r.graph_rebuilt)
    report = {
        "pilot": pilot.summary(),
        "requested_hashes": requested,
        "hashes_matched_in_bigquery": matched,
        "trees_recovered_complete": recovered,
        "defiguard_graphs_unblocked": unblocked,
        "defiguard_graphs_unblocked_of_requested": f"{unblocked}/{requested}",
        "reverted_ancestry_resolved": sum(1 for r in rebuilds if r.ancestry_resolved),
        "blockscan_preprocess_ok": sum(1 for r in rebuilds if r.blockscan_ok),
        "blockscan_preprocess_attempted": sum(1 for r in rebuilds
                                              if r.blockscan_ok is not None),
        "missing_reasons": _count(r.missing_reason for r in rebuilds if r.missing_reason),
        "tree_issues": _count(i["issue"] for r in rebuilds if r.tree is not None
                              for i in r.tree.issues),
        "root_cross_check": {
            "agreed": sum(1 for r in rebuilds if r.root_check.get("agrees")),
            "disagreed": sum(1 for r in rebuilds if r.root_check.get("mismatches")),
            "not_checked": sum(1 for r in rebuilds if not r.root_check.get("checked")),
        },
        **extras,
    }
    report_path = out_dir / "trace_backfill_report.json"
    report_path.write_bytes(canonical_json(report))
    return {"coverage_table": coverage_path, "report": report_path}


def _count(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
