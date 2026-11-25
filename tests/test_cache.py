"""The cache: neutral paths, provenance, resume, and separate failure states."""
from __future__ import annotations

import pytest

from txbench.cache import (CacheRef, CacheStore, FAILED_RETRYABLE, OK, OVER_CAP, UNSUPPORTED,
                           TARGET_BLOCK)
from txbench.config import M_RECEIPT, M_TX
from txbench.errors import CacheError

HASH = "0x" + "11" * 32


def test_cache_path_is_neutral_and_keyed_by_everything_that_matters():
    ref = CacheRef.for_tx(1, HASH, M_TX)
    path = str(ref.relative_path())
    assert "chain=1" in path and M_TX in path and "tracer=none" in path and "/v1/" in path
    for word in ("attack", "benign", "exploit", "label", "negative"):
        assert word not in path.lower(), "a cache path must not leak a class"


def test_put_then_hit_records_provenance(tmp_path):
    store = CacheStore(tmp_path)
    ref = CacheRef.for_tx(1, HASH, M_TX)
    entry = store.put(ref, {"hash": HASH}, provider="fake", block_number=99,
                      block_hash="0xabc", tx_index=4, units_charged=20)
    assert entry.state == OK
    assert entry.provider == "fake" and entry.block_number == 99 and entry.tx_index == 4
    assert entry.content_sha256 and entry.payload_bytes
    assert store.hit(ref) is not None
    assert store.load(ref) == {"hash": HASH}


def test_a_second_run_finds_a_hit_and_needs_no_call(tmp_path):
    store = CacheStore(tmp_path)
    ref = CacheRef.for_tx(1, HASH, M_RECEIPT)
    store.put(ref, {"logs": []}, provider="fake")
    reopened = CacheStore(tmp_path)
    assert reopened.hit(ref) is not None, "a resumed run must reuse a completed request"


def test_corrupted_payload_is_detected(tmp_path):
    import gzip
    store = CacheStore(tmp_path)
    ref = CacheRef.for_tx(1, HASH, M_TX)
    entry = store.put(ref, {"hash": HASH}, provider="fake")
    with gzip.open(entry.blob_path, "wb") as f:
        f.write(b'{"hash": "0xdifferent"}')
    with pytest.raises(CacheError, match="checksum"):
        store.read_payload(store.get(ref))


def test_failure_states_are_persisted_separately(tmp_path):
    store = CacheStore(tmp_path)
    retryable = CacheRef.for_tx(1, HASH, M_TX)
    permanent = CacheRef.for_tx(1, HASH, "debug_traceTransaction")
    store.record_failure(retryable, state=FAILED_RETRYABLE, provider="fake", error="timeout")
    store.record_failure(permanent, state=UNSUPPORTED, provider="fake", error="not supported")
    assert store.get(retryable).is_hit is False
    assert store.get(retryable).is_permanent_miss is False, "a timeout is worth retrying"
    assert store.get(permanent).is_permanent_miss is True, "an unsupported method is not"
    states = {e.ref.method: e.state for e in store.iter_entries()}
    assert states == {M_TX: FAILED_RETRYABLE, "debug_traceTransaction": UNSUPPORTED}


def test_attempts_accumulate_across_failures(tmp_path):
    store = CacheStore(tmp_path)
    ref = CacheRef.for_tx(1, HASH, M_TX)
    store.record_failure(ref, state=FAILED_RETRYABLE, provider="fake", error="first")
    store.record_failure(ref, state=FAILED_RETRYABLE, provider="fake", error="second")
    assert store.get(ref).attempts == 2


def test_over_cap_payload_is_recorded_but_not_stored(tmp_path):
    store = CacheStore(tmp_path)
    ref = CacheRef.for_tx(1, HASH, "debug_traceTransaction")
    entry = store.put(ref, {"big": "x" * 1000}, provider="fake", max_payload_bytes=10)
    assert entry.state == OVER_CAP
    assert entry.blob_path is None
    assert entry.payload_bytes > 10
    assert entry.completeness == "truncated_not_stored"


def test_credentials_never_reach_the_cache_index(tmp_path):
    from txbench.credentials import register_secret
    register_secret("supersecretkey-abcdefgh")
    store = CacheStore(tmp_path)
    ref = CacheRef.for_tx(1, HASH, M_TX)
    store.record_failure(ref, state=FAILED_RETRYABLE, provider="fake",
                         error="GET https://x.invalid/v2/supersecretkey-abcdefgh failed")
    assert "supersecretkey" not in (store.get(ref).last_error or "")


def test_block_entries_are_shared_across_transactions(tmp_path):
    store = CacheStore(tmp_path)
    ref = CacheRef.for_block(1, 18_000_000, "eth_getBlockByNumber")
    assert ref.target_kind == TARGET_BLOCK
    store.put(ref, {"number": "0x112a880"}, provider="fake")
    again = CacheRef.for_block(1, 18_000_000, "eth_getBlockByNumber")
    assert store.hit(again) is not None, "one header serves every tx in that block"
