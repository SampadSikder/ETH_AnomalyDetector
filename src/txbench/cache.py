"""The raw-response cache: the thing that makes a second run cost nothing.

Design rules this file enforces:
  * paths are **neutral** - keyed by chain, target, method, tracer config and
    cache schema version, never by class, so a cache path cannot leak a label;
  * every stored payload carries provenance: provider, content checksum, byte
    count, completeness, chain location and collection time;
  * writes are atomic (temp file + rename) and payloads are gzip-compressed;
  * retryable failures, permanently unsupported methods and over-cap
    (truncated) payloads are persisted *separately* from successes, so a
    resumed run retries only what is worth retrying and re-requests nothing
    that already succeeded.
"""
from __future__ import annotations

import gzip
import json
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import CACHE_SCHEMA_VERSION
from .errors import CacheError
from .hashutil import sha256_bytes
from .ids import normalize_chain_id, normalize_tx_hash
from .logs import get_logger
from .credentials import redact

log = get_logger("cache")

# entry states
OK = "ok"
FAILED_RETRYABLE = "failed_retryable"
UNSUPPORTED = "unsupported"          # permanent: the plan/provider lacks the method
OVER_CAP = "over_cap"                # payload exceeded the stored-bytes cap
INVALID = "invalid"                  # a response that failed validation

TARGET_TX = "tx"
TARGET_BLOCK = "block"

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    cache_key      TEXT PRIMARY KEY,
    chain_id       INTEGER NOT NULL,
    target_kind    TEXT NOT NULL,
    target_id      TEXT NOT NULL,
    method         TEXT NOT NULL,
    tracer_config  TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    state          TEXT NOT NULL,
    blob_path      TEXT,
    provider       TEXT,
    content_sha256 TEXT,
    payload_bytes  INTEGER,
    completeness   TEXT,
    block_number   INTEGER,
    block_hash     TEXT,
    tx_index       INTEGER,
    units_charged  INTEGER DEFAULT 0,
    attempts       INTEGER DEFAULT 0,
    last_error     TEXT,
    collected_utc  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS entries_target ON entries (chain_id, target_kind, target_id);
CREATE INDEX IF NOT EXISTS entries_state ON entries (state, method);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class CacheRef:
    """Everything that identifies one cached request."""
    chain_id: int
    target_kind: str
    target_id: str
    method: str
    tracer_config: str = "none"
    schema_version: int = CACHE_SCHEMA_VERSION

    @staticmethod
    def for_tx(chain_id: int, tx_hash: str, method: str, tracer_config: str = "none") -> "CacheRef":
        return CacheRef(normalize_chain_id(chain_id), TARGET_TX, normalize_tx_hash(tx_hash),
                        method, tracer_config)

    @staticmethod
    def for_block(chain_id: int, block_number: int, method: str,
                  tracer_config: str = "none") -> "CacheRef":
        return CacheRef(normalize_chain_id(chain_id), TARGET_BLOCK, str(int(block_number)),
                        method, tracer_config)

    @property
    def cache_key(self) -> str:
        return "|".join((str(self.chain_id), self.target_kind, self.target_id, self.method,
                         self.tracer_config, f"v{self.schema_version}"))

    def relative_path(self) -> Path:
        """A neutral, shardable path. Carries no class information."""
        if self.target_kind == TARGET_TX:
            shard = self.target_id[2:4]
            name = f"{self.target_id}.json.gz"
        else:
            bucket = int(self.target_id) // 100_000
            shard = f"{bucket:05d}"
            name = f"{int(self.target_id):09d}.json.gz"
        return Path(f"chain={self.chain_id}") / self.method / f"tracer={self.tracer_config}" / \
            f"v{self.schema_version}" / shard / name


@dataclass
class CacheEntry:
    ref: CacheRef
    state: str
    blob_path: Path | None
    provider: str | None
    content_sha256: str | None
    payload_bytes: int | None
    completeness: str | None
    block_number: int | None
    block_hash: str | None
    tx_index: int | None
    units_charged: int
    attempts: int
    last_error: str | None
    collected_utc: str

    @property
    def is_hit(self) -> bool:
        """A hit is a stored success. Nothing else spares a call."""
        return self.state == OK and self.blob_path is not None

    @property
    def is_permanent_miss(self) -> bool:
        return self.state in (UNSUPPORTED, OVER_CAP)


class CacheStore:
    """SQLite index over gzip blobs. Safe to open repeatedly; never deletes."""

    def __init__(self, root: Path, index_path: Path | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = Path(index_path) if index_path else self.root / "index.sqlite"
        self._conn = sqlite3.connect(self.index_path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        with closing(self._conn.cursor()) as cur:
            cur.executescript(SCHEMA)
            self._conn.commit()

    # --- reads ---------------------------------------------------------------
    def get(self, ref: CacheRef) -> CacheEntry | None:
        row = self._conn.execute("SELECT * FROM entries WHERE cache_key = ?",
                                 (ref.cache_key,)).fetchone()
        return self._row_to_entry(row) if row else None

    def hit(self, ref: CacheRef) -> CacheEntry | None:
        entry = self.get(ref)
        if entry and entry.is_hit and entry.blob_path and entry.blob_path.is_file():
            return entry
        return None

    def read_payload(self, entry: CacheEntry) -> Any:
        if not entry.blob_path:
            raise CacheError(f"cache entry {entry.ref.cache_key} has no stored payload "
                             f"(state={entry.state})")
        with gzip.open(entry.blob_path, "rb") as f:
            data = f.read()
        if entry.content_sha256 and sha256_bytes(data) != entry.content_sha256:
            raise CacheError(f"cached payload for {entry.ref.cache_key} does not match its "
                             "recorded checksum; delete the blob and refetch")
        return json.loads(data.decode("utf-8"))

    def load(self, ref: CacheRef) -> Any | None:
        entry = self.hit(ref)
        return self.read_payload(entry) if entry else None

    def iter_entries(self, *, state: str | None = None, method: str | None = None
                     ) -> Iterator[CacheEntry]:
        sql = "SELECT * FROM entries"
        clauses, args = [], []
        if state:
            clauses.append("state = ?")
            args.append(state)
        if method:
            clauses.append("method = ?")
            args.append(method)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY cache_key"
        for row in self._conn.execute(sql, args):
            yield self._row_to_entry(row)

    def stats(self) -> dict[str, Any]:
        rows = self._conn.execute(
            "SELECT method, state, COUNT(*) n, COALESCE(SUM(payload_bytes),0) b, "
            "COALESCE(SUM(units_charged),0) u FROM entries GROUP BY method, state "
            "ORDER BY method, state").fetchall()
        by_method: dict[str, dict[str, Any]] = {}
        totals = {"entries": 0, "payload_bytes": 0, "units_charged": 0}
        for row in rows:
            slot = by_method.setdefault(row["method"], {})
            slot[row["state"]] = {"entries": row["n"], "payload_bytes": row["b"],
                                  "units_charged": row["u"]}
            totals["entries"] += row["n"]
            totals["payload_bytes"] += row["b"]
            totals["units_charged"] += row["u"]
        return {"index": str(self.index_path), "totals": totals, "by_method": by_method}

    # --- writes --------------------------------------------------------------
    def put(self, ref: CacheRef, payload: Any, *, provider: str, raw_bytes: bytes | None = None,
            completeness: str = "complete", block_number: int | None = None,
            block_hash: str | None = None, tx_index: int | None = None,
            units_charged: int = 0, max_payload_bytes: int | None = None) -> CacheEntry:
        """Store a successful response. Atomic; keeps the exact bytes received."""
        data = raw_bytes if raw_bytes is not None else json.dumps(
            payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        size = len(data)
        if max_payload_bytes is not None and size > max_payload_bytes:
            log.warning("payload for %s is %d bytes, over the %d-byte cap: recorded as %s, "
                        "not stored", ref.cache_key, size, max_payload_bytes, OVER_CAP)
            return self._record(ref, OVER_CAP, blob_path=None, provider=provider,
                                content_sha256=sha256_bytes(data), payload_bytes=size,
                                completeness="truncated_not_stored", block_number=block_number,
                                block_hash=block_hash, tx_index=tx_index,
                                units_charged=units_charged, error=None)
        blob = self.root / ref.relative_path()
        blob.parent.mkdir(parents=True, exist_ok=True)
        tmp = blob.with_suffix(blob.suffix + f".tmp{os.getpid()}")
        # mtime=0 so the compressed bytes are reproducible run to run.
        with open(tmp, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6,
                                                   mtime=0) as f:
            f.write(data)
        os.replace(tmp, blob)
        return self._record(ref, OK, blob_path=blob, provider=provider,
                            content_sha256=sha256_bytes(data), payload_bytes=size,
                            completeness=completeness, block_number=block_number,
                            block_hash=block_hash, tx_index=tx_index,
                            units_charged=units_charged, error=None)

    def record_failure(self, ref: CacheRef, *, state: str, provider: str | None,
                       error: str, units_charged: int = 0) -> CacheEntry:
        if state not in (FAILED_RETRYABLE, UNSUPPORTED, INVALID):
            raise CacheError(f"not a failure state: {state!r}")
        previous = self.get(ref)
        attempts = (previous.attempts if previous else 0) + 1
        return self._record(ref, state, blob_path=None, provider=provider, content_sha256=None,
                            payload_bytes=None, completeness=None, block_number=None,
                            block_hash=None, tx_index=None, units_charged=units_charged,
                            error=error, attempts=attempts)

    def _record(self, ref: CacheRef, state: str, *, blob_path: Path | None, provider: str | None,
                content_sha256: str | None, payload_bytes: int | None, completeness: str | None,
                block_number: int | None, block_hash: str | None, tx_index: int | None,
                units_charged: int, error: str | None, attempts: int | None = None) -> CacheEntry:
        existing = self.get(ref)
        if attempts is None:
            attempts = (existing.attempts if existing else 0) + 1
        charged = (existing.units_charged if existing else 0) + int(units_charged)
        row = {
            "cache_key": ref.cache_key, "chain_id": ref.chain_id,
            "target_kind": ref.target_kind, "target_id": ref.target_id, "method": ref.method,
            "tracer_config": ref.tracer_config, "schema_version": ref.schema_version,
            "state": state, "blob_path": str(blob_path) if blob_path else None,
            "provider": provider, "content_sha256": content_sha256,
            "payload_bytes": payload_bytes, "completeness": completeness,
            "block_number": block_number, "block_hash": block_hash, "tx_index": tx_index,
            "units_charged": charged, "attempts": attempts,
            "last_error": redact(error)[:500] if error else None,
            "collected_utc": utc_now(),
        }
        cols = ",".join(row)
        placeholders = ",".join(f":{c}" for c in row)
        updates = ",".join(f"{c}=excluded.{c}" for c in row if c != "cache_key")
        self._conn.execute(
            f"INSERT INTO entries ({cols}) VALUES ({placeholders}) "
            f"ON CONFLICT(cache_key) DO UPDATE SET {updates}", row)
        self._conn.commit()
        return self._row_to_entry(
            self._conn.execute("SELECT * FROM entries WHERE cache_key = ?",
                               (ref.cache_key,)).fetchone())

    def close(self) -> None:
        self._conn.close()

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        ref = CacheRef(chain_id=row["chain_id"], target_kind=row["target_kind"],
                       target_id=row["target_id"], method=row["method"],
                       tracer_config=row["tracer_config"], schema_version=row["schema_version"])
        return CacheEntry(
            ref=ref, state=row["state"],
            blob_path=Path(row["blob_path"]) if row["blob_path"] else None,
            provider=row["provider"], content_sha256=row["content_sha256"],
            payload_bytes=row["payload_bytes"], completeness=row["completeness"],
            block_number=row["block_number"], block_hash=row["block_hash"],
            tx_index=row["tx_index"], units_charged=row["units_charged"] or 0,
            attempts=row["attempts"] or 0, last_error=row["last_error"],
            collected_utc=row["collected_utc"])


def write_cache_index_report(store: CacheStore, out_path: Path) -> Path:
    """A reviewable index of the cache: one row per request, no payloads."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for e in store.iter_entries():
        rows.append({
            "cache_key": e.ref.cache_key, "chain_id": e.ref.chain_id,
            "target_kind": e.ref.target_kind, "target_id": e.ref.target_id,
            "method": e.ref.method, "tracer_config": e.ref.tracer_config,
            "schema_version": e.ref.schema_version, "state": e.state,
            "provider": e.provider, "content_sha256": e.content_sha256,
            "payload_bytes": e.payload_bytes, "completeness": e.completeness,
            "block_number": e.block_number, "block_hash": e.block_hash,
            "tx_index": e.tx_index, "units_charged": e.units_charged,
            "attempts": e.attempts, "last_error": e.last_error,
            "collected_utc": e.collected_utc,
            "blob_path": str(e.blob_path) if e.blob_path else None,
        })
    import pandas as pd
    df = pd.DataFrame(rows, columns=[
        "cache_key", "chain_id", "target_kind", "target_id", "method", "tracer_config",
        "schema_version", "state", "provider", "content_sha256", "payload_bytes",
        "completeness", "block_number", "block_hash", "tx_index", "units_charged",
        "attempts", "last_error", "collected_utc", "blob_path"])
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)
    return out_path
