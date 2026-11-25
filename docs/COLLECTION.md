# The collector

## Order of operations

The collector's first job is not to fetch. It is:

1. **inventory** what the upstream release already supplies, per transaction
   (`src/txbench/inventory.py`);
2. **consult the cache**;
3. only then **plan** provider calls, and **price the plan before making it**;
4. **fetch** under per-provider limits, bounded retries and a budget fuse;
5. **reconcile** every fetched record against the upstream one, and store it with
   provenance.

## What this cost on the real cohort

| scenario | transactions | calls | Alchemy CU | share of the 30M monthly free allowance |
|---|---|---|---|---|
| **base** — fetch only what upstream lacks | 1,632 | **0** | **0** | 0 |
| cross-check — re-read envelope + receipt for every row | 1,632 | 3,264 | 65,280 | 0.22 % |

The release supplies a complete envelope, a receipt and a complete call tree for
all 1,632 transactions, so the base pass needs no provider at all. The
cross-check scenario reproduces the budgeted figure exactly — one transaction
lookup and one receipt lookup each at 20 CU — and is available with
`bench collect --dry-run --verify-upstream`. It excludes headers, capability
probes, retries and richer traces, as the budget note says.

Unique block headers needed: 0 — the release carries `block_hash` and
`block_timestamp` on every record. A header would otherwise be fetched once and
shared across every transaction in that block.

## What is genuinely missing, and where it would come from

| data | status on this cohort | route |
|---|---|---|
| envelope, top-level calldata | supplied for 1,632/1,632 | upstream; else `eth_getTransactionByHash` |
| receipt, outcome, emitting addresses, ordered topics, log data | supplied for 1,632/1,632 | upstream; else `eth_getTransactionReceipt` |
| block time and canonical block identity | supplied for 1,632/1,632 | upstream; else one header per block, shared |
| internal calls, nested calldata, return bytes | **frames supplied, nesting absent** | a call-tracer export, a Parity-style trace with `trace_address`, or the BigQuery traces table |
| ordered EVM steps, state reads/writes, stack/memory operands | absent | a struct-log trace from an archive node, or an explicitly authorised trace-capable provider |

**The nesting gap.** `internal_transactions` is an ordered flat list whose frames
carry `type`, `from`, `to`, `value`, `gas`, `gas_used`, `input`, `output` and
`error` — and no `trace_address`, parent pointer or depth. Depth is not
recoverable from a flat list, because one address can appear in many frames.
This project records the gap (`field_gaps["call_tree_nesting"]`) instead of
inventing a tree, and `tests/test_real_release.py` asserts it on real data.

One concrete free backfill candidate is the public BigQuery table
`bigquery-public-data.goog_blockchain_ethereum_mainnet_us.traces`, which Google
documents as Parity-style action traces — those carry a trace address, so they
could recover nesting and nested calldata. It is **not** wired in: using it means
inspecting the schema and its historical coverage, matching hashes, restricting
date partitions, dry-running bytes scanned and setting a maximum-bytes-billed
limit inside the free allowance, and only proceeding if that is already
accessible within budget. `google-cloud-bigquery` is an optional extra and is
deliberately not installed by default. Opcode steps would still need a different
source.

## Providers and limits

Configured in `config/benchmark.toml`; the effective settings are logged on every
run and written into the cost report.

| | Alchemy (primary) | dRPC (fallback) |
|---|---|---|
| unit | CU | request |
| documented free allowance | 30M CU/month, 300 CU/s | ~10M equivalent multichain requests |
| configured rate limit | 300 CU/s | 30 req/s (conservative; the vendor's comparison assigns 100 RPS to PAYG) |
| per-run fuse | 400,000 CU | 20,000 requests |
| Debug / Trace methods | **disabled** (excluded from the Free plan) | **disabled** (paid on Ethereum) |

No paid plan is ever activated: methods a plan excludes are configured off rather
than attempted. Account plans differ, so every limit is overridable and whatever
is set is what gets logged and reported.

Behaviour under stress, all covered by `tests/test_rpc.py`:

- **429 / throttling** — retried with exponential backoff plus jitter, honouring
  `Retry-After`, bounded by `max_retries`;
- **"method not supported"** — a *permanent* answer: the method is disabled for
  the rest of the run rather than retried, and the attempt is recorded as
  `unsupported`, not as a failure to retry;
- **partial or malformed replies** — a reply with neither `result` nor `error` is
  rejected; a truncated body fails loudly;
- **batches** — members are counted and charged individually;
- **budget fuses** — reaching the call cap or a per-provider unit budget raises
  and stops the run cleanly, naming the setting to raise;
- **fallback** — a provider is tried for a method only after the primary fails or
  would exceed its budget, and only for methods it is allowed to serve.

A capability probe assesses each provider on the same small set of historical
hashes — including a `eth_chainId` check against the expected chain — before it
is trusted with a full pass (`data/collection/provider_probe.json`).

## The cache

`data/cache/` with a SQLite index at `data/cache/index.sqlite`.

- Paths are **neutral**: `chain=1/<method>/tracer=<cfg>/v<schema>/<shard>/<hash>.json.gz`.
  A cache path cannot leak a class, and this is asserted in `tests/test_cache.py`.
- Keyed by chain, target, method, tracer configuration and cache schema version.
- Every entry records provider, content SHA-256, byte count, completeness, block
  number, block hash, transaction index, units charged, attempts and collection
  time.
- Payloads are gzip-compressed with `mtime=0` so the bytes are reproducible, and
  written atomically (temp file + rename). A payload that fails its recorded
  checksum on read is refused.
- Retryable failures, permanently unsupported methods and over-cap payloads are
  persisted **separately**, so a resumed run retries only what is worth retrying
  and re-requests nothing that already succeeded. A payload over
  `max_stored_payload_bytes` is recorded with its size and marked
  `truncated_not_stored` rather than silently clipped.
- A resumed run reuses every completed request and preserves all manifest rows;
  a second completed run makes zero data calls.

Fetched records are cross-checked before they are stored: the reply must be for
the requested transaction or block, and the chain id, hash, block and
transaction index are reconciled against the upstream record field by field,
with any mismatch reported in the collection report.

## The pilot

`bench collect --pilot 20 --seed 42` selects deterministically: strata over
`(label, block year, trace-size band)`, round-robin across strata, with a seeded
shuffle inside each. The same manifest, size and seed always give the same
hashes, and both classes, several periods and both short and long traces appear
before any stratum repeats. The chosen keys are written to
`data/collection/pilot_keys.json`, and `bench audit-inputs --pilot` reuses them.
Pilot records are reused by the full run through the same cache.
