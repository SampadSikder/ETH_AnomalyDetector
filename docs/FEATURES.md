# The two feature views

Two views exist so a missing historical call tree cannot silently change the
experiment. Both are exported for the FT-Transformer, and **coverage is reported
per view, separately.**

| view | one row per transaction | coverage on the real cohort | vocabulary |
|---|---|---|---|
| `tx-receipt` | the top-level selector candidate, plus one count per distinct receipt-log first topic | 1,632 / 1,632 | 690 tokens |
| `full-calls-receipt` | one count per distinct selector candidate over **every** call frame, the outer call included **exactly once**, plus the same receipt-topic counts | 1,632 / 1,632 | 2,096 tokens |

Valid when a receipt exists, for view 1. View 2 additionally requires a call tree
the source asserts complete, with input bytes on every frame — which is why the
release's own trace status is read rather than assumed.

## What a token is, exactly

**`sel:<4 bytes>`** — the first four bytes of a *normal call's* input, when that
input is at least four bytes long. Four bytes of calldata are a *candidate*
function id and nothing more: nothing here verifies them against an ABI, so they
stay categorical hex values and are never renamed to a function signature.

**`tpc0:<32 bytes>`** — a receipt log's **observed** first topic, stored as
`raw_topic0`. A non-anonymous ABI event ordinarily puts its signature hash in
that slot, but an anonymous event may use it for an argument, and zero-topic logs
exist, so no event signature is claimed from the bytes alone. The release's own
`decoded_event` is carried alongside as *provenance*, never as evidence.

**`pre:<address>`** — a call whose target is a known precompile (`0x01`–`0x11`).
A precompile takes raw arguments, not an ABI-encoded call, so its first four
bytes are arbitrary data. They are kept (in `candidate_bytes`) but deliberately
held **out of the selector namespace**, so arbitrary bytes are never presented as
verified ABI function ids.

Selectors, precompile targets and topics never share a vocabulary slot.

## Explicit statuses instead of silent absence

| status | meaning |
|---|---|
| `ok` | a selector was extracted |
| `empty_input` | zero bytes of calldata, e.g. a plain value transfer |
| `short_input` | one to three bytes: not a selector |
| `contract_creation` | initcode, not a function call |
| `precompile_target` | argument bytes, not a function id |
| `not_a_call` | a frame type that carries no calldata |
| `unavailable` | the source supplied no input bytes for this frame |

Call type is recorded per frame, and the per-frame table keeps raw call order,
parent paths where a source supplies them, reverts and logs **outside** the
tabular view, for the other tools.

## Missing versus zero

> Unavailable data stays missing. Only an **observed absence** receives a zero.

- A receipt that exists with an empty log list is a valid, common outcome: its
  topic counts are genuine zeros.
- A transaction in which no code executed (the release's
  `no_code_executed`, 268 negatives) genuinely has no internal calls: zero is
  correct.
- A row that is not usable in a view is **not** emitted as zeros. It is emitted
  with `usable = false`, a stated `unusable_reason`, and **null** across every
  feature column — and it stays in the denominator.

Attempted calls are counted separately from committed receipt events
(`attempted_calls` vs `committed_receipt_events` in `row_status.parquet`), and
reverted frames are counted on their own.

## The pipeline, in order

1. **Sparse counts first.** `tokens.parquet` holds `(tx_key, token, count,
   namespace)` long-form rows, written before any vocabulary exists.
2. **Vocabulary fitted on training rows only.** `vocab_<split>.json` records the
   split, the frequency cutoff, the number of training rows it saw and a digest
   of the exact keys used. A token that appears only in validation or test
   cannot enter it — asserted in `tests/test_features.py`.
3. **Matrices per partition.** `matrix_<split>_<train|validation|test>.parquet`,
   with out-of-vocabulary mass preserved in `oov_sel_count`, `oov_tpc0_count`
   and `oov_pre_count` rather than dropped.
4. **Scalers fitted on training rows only**, persisted (`scalers_<split>.json`)
   rather than applied, so a later split cannot refit them by accident.
5. **Labels joined by key** for supervised training. `label` is the target;
   `tx_key` is the key. Identifiers, label evidence, attack categories, source
   tiers, collection paths and detector success flags are excluded from the
   feature columns, and the exclusion list is in the feature dictionary.

Equal train/test column definitions within each view follow from a single
vocabulary per view and split. Each view's `feature_dictionary.json` carries
every definition above, the partition sizes and the coverage breakdown.
