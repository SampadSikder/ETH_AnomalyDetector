"""Assemble normalised transactions from the upstream release and the cache.

Precedence is deliberate and documented: **valid upstream fields first**, and
a provider read only fills what upstream did not supply. That is what makes a
second run free, and it keeps the authoritative copy of a field in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from .cache import CacheRef, CacheStore
from .config import M_BLOCK_BY_NUMBER, M_RECEIPT, M_TX
from .inventory import normalized_from_upstream
from .logs import get_logger
from .trace_model import NormalizedTx, from_rpc, merge_rpc_into

log = get_logger("assemble")


def _cached(store: CacheStore | None, chain_id: int, method: str, tx_hash: str) -> Any | None:
    if store is None:
        return None
    return store.load(CacheRef.for_tx(chain_id, tx_hash, method))


def _cached_block(store: CacheStore | None, chain_id: int, block_number: int | None) -> Any | None:
    if store is None or block_number is None:
        return None
    return store.load(CacheRef.for_block(chain_id, block_number, M_BLOCK_BY_NUMBER))


def normalized_for(tx_key: str, chain_id: int, tx_hash: str, *, traces_dir: Path | None,
                   store: CacheStore | None, trace_status: str | None = None
                   ) -> NormalizedTx | None:
    """One record, upstream first, cache second."""
    tx = None
    if traces_dir is not None:
        tx, _ = normalized_from_upstream(traces_dir, chain_id, tx_hash, trace_status)
    tx_payload = _cached(store, chain_id, M_TX, tx_hash)
    receipt_payload = _cached(store, chain_id, M_RECEIPT, tx_hash)
    if tx is None:
        if tx_payload is None and receipt_payload is None:
            return None
        tx = from_rpc(tx_payload, receipt_payload, chain_id=chain_id, tx_hash=tx_hash,
                      block_payload=None)
        block_payload = _cached_block(store, chain_id, tx.block_number)
        return merge_rpc_into(tx, None, None, block_payload)
    block_payload = _cached_block(store, chain_id, tx.block_number)
    return merge_rpc_into(tx, tx_payload, receipt_payload, block_payload)


def records(manifest: pd.DataFrame, *, traces_dir: Path | None, store: CacheStore | None,
            trace_status: dict[str, dict[str, Any]] | None = None,
            keys: Iterable[str] | None = None) -> Iterator[tuple[str, NormalizedTx | None]]:
    wanted = set(keys) if keys is not None else None
    statuses = trace_status or {}
    frame = manifest if wanted is None else manifest[manifest["tx_key"].isin(wanted)]
    for row in frame.sort_values("tx_key").itertuples(index=False):
        status = (statuses.get(row.tx_hash) or {}).get("status")
        yield row.tx_key, normalized_for(row.tx_key, int(row.chain_id), row.tx_hash,
                                         traces_dir=traces_dir, store=store,
                                         trace_status=status)


def labels_of(manifest: pd.DataFrame) -> dict[str, int]:
    return {row.tx_key: int(row.label) for row in manifest.itertuples(index=False)}


def incidents_of(manifest: pd.DataFrame) -> dict[str, str]:
    return {row.tx_key: (row.group_incident or row.tx_key)
            for row in manifest.itertuples(index=False)}
