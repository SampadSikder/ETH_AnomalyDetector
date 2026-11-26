# The transaction set

The transaction set is the fixed list of transactions this benchmark runs on:
1,632 Ethereum mainnet transactions chosen out of the release's 3,036, with
nothing added or removed afterwards. The fingerprint below pins that exact list,
so any later run can prove it used the same transactions.

## Source

| | |
|---|---|
| Release | `TxBench-v1.0.1` |
| DOI | [10.5281/zenodo.23758032](https://doi.org/10.5281/zenodo.23758032) |
| Archive | `TxBench-v1.0.1.zip`, 2,855,342,387 bytes, md5 `1e57920ae8e19b6245e7548ad4e5bd9d` (verified against the record on download) |
| Access | Open. The paper describes restricted archive files; the published record is in fact openly downloadable, which is what `bench fetch-release` uses. |
| Files read | 18 roles resolved, 0 missing; every one recorded with its SHA-256 in `data/manifest/release_provenance.json` |

## Row counts, before and after every filter

| filter | rows in | rows out | dropped |
|---|---|---|---|
| read_source | 3,036 | 3,036 | 0 |
| validate_ids_and_labels | 3,036 | 3,036 | 0 |
| chain_id_eq_1 | 3,036 | 1,632 | 1,404 |
| collapse_duplicates | 1,632 | 1,632 | 0 |

Dropped by chain: BSC 927, Base 195, Arbitrum 129, Optimism 75, Polygon 51,
Avalanche 27. Those transactions are removed here, not kept aside for later: no
chain other than Ethereum appears anywhere in the set, including in the training
rows. Each chain's total is exactly three times its accepted-attack count
because the release pairs every attack with two negatives sampled on the same
chain, so a chain leaves as one whole block. Ethereum keeps that same 1:2 ratio,
which is why the filter changes no label balance: 33.3% attacks before and
after.

**Result: 1,632 rows — 544 attacks and 1,088 negatives.** This matches the
paper's v1 cross-check for Ethereum exactly (544 attacks, 1,088 negatives before
key compromises are separated), so no discrepancy needed investigating. The
counts are not hard-coded anywhere; `tests/test_real_release.py` asserts them
against the pinned release, and the chain filter reports its own arithmetic.

Attack classes, verbatim from the release's `category` column:
**exploit 386**, **key_compromise 158**. Benign transactions by category:
behavioral_hard 465, financial_hard 279, ordinary 186, deployment 158 — all
`tier1_verified`.

## Identity and integrity rules

- Every transaction is keyed by `(chain_id, lower-case 0x-prefixed 32-byte hash)`,
  written `tx_key = "1:0x…"`. Nothing downstream keys on a bare hash.
- A chain slug that disagrees with its `chain_id` quarantines the row rather than
  trusting either column.
- **Identical duplicates** collapse into one row, with every source row number
  kept in `source_rows` and `duplicate_count` recording how many there were.
- **Conflicting duplicates** — the same hash with different identity fields — are
  quarantined in full. Neither copy enters the manifest.
- An unrecognised label, an invalid hash, an invalid chain id or an undocumented
  attack class is quarantined with a reason. **None of these ever becomes a
  negative.** The same rule holds later for RPC failures and reverted calls.
- Labels apply to whole transactions and are imported exactly as supplied; no
  label is inferred, re-derived or adjusted.

The fixture set exercises each of these paths; see `tests/test_importer.py`.

## Reproducibility

The manifest is immutable: `bench import-txbench` writes it atomically and never
edits in place. Its identity is pinned by a digest over the sorted
`(tx_key, label)` pairs:

```
cohort_sha256 = b07e892a606422dff1fd280bcd940c60f400fcba67f6eb3231084c02f4d81fcf
```

Deliberately narrow — it does not move when an unrelated column is added, and it
does move when a hash or a label changes (`tests/test_importer.py`). The same
source files and config produce the same sorted hashes and labels.

## Schema mapping

Written only after reading the release's `step4_benign/docs/data_dictionary.md`.
The resolved mapping is persisted at `data/manifest/schema_mapping.json`, so an
auditor can see which upstream column became which field, and an upstream schema
change fails loudly instead of silently shifting a label.

| this project | upstream column |
|---|---|
| `tx_hash`, `chain_id`, `label` | `tx_hash`, `chain_id`, `label` (required) |
| `source_category` | `category` — the attack class for an attack, the primary category for a negative |
| `attack_class` | `category`, kept only when the row is an attack and the value is one of `exploit` / `key_compromise` |
| `benign_category` | `category`, kept only when the row is a negative |
| `label_tier` | `label_confidence` (`oracle_accepted` for attacks, tier for negatives) |
| `group_incident`, `group_protocol`, `group_address` | the same three columns |
| `block_number`, `block_timestamp` | the same, the transaction's chain location |
| `split_*` | every upstream split column, imported verbatim |

## Trace completeness

The release assesses each enriched trace itself, in
`step2_traces/output/traces_status.json` (attacks) and
`step4_benign/output/traces_status.json` (negatives). Reading those files is what
lets this project call a call tree **complete** rather than merely **present**.

All 1,632 Ethereum rows are `status: ok`. `internal_calls_source` is `rpc_trace`
for all 544 attacks and for 820 negatives, and `no_code_executed` for 268
negatives — the latter meaning no internal calls happened, an **observed
absence**, so zero is the correct value there rather than "missing".

## What is kept separate

Upstream files stay under `data/upstream/` and are never rewritten. Model
features are derived artifacts under `data/features/`. Identifiers, label
evidence, attack categories, source tiers, collection paths and detector success
flags are excluded from the feature matrices by construction, and the exclusion
list is asserted in `tests/test_features.py`.

## Licensing and redistribution

From the release's own `LICENSE` and `DATASET_LICENSE.md`, recorded in
`data/manifest/release_provenance.json`:

- pipeline code: **MIT**;
- TxBench's own material (labels, validation and review decisions, SCWE
  readings, the selection and arrangement of records): **CC BY 4.0**;
- third-party material keeps its source's terms — the BlockSec Security
  Incidents Library export and labels derived from it, the EVM Hack Registry and
  its proofs of concept (licensed per item), DeFiLlama price data, and the OWASP
  SCWE snapshot (**CC BY-SA 4.0**).

Because third-party terms vary per item, **this repository redistributes no
upstream bytes.** `data/upstream/` is git-ignored and is rebuilt from the DOI by
`bench fetch-release`. Review `DATASET_LICENSE.md` sections 2 and 3 before
packaging any upstream data for distribution. Cite the paper and the record's
DOI when using this transaction set.
