# Evaluation

## Splits

Every upstream split column is preserved verbatim. Each one is audited against
**the promise it actually makes**: overlap in a group a split never claimed to
hold apart is by design, not a leak — a temporal split cuts by date, so the same
protocol or sender appears on both sides, which is precisely why the protocol and
address splits exist. Overlap is reported for every group column either way; only
a broken promise counts as a conflict.

| split | promises to keep whole | result on this transaction set |
|---|---|---|
| `split_temporal` | `group_incident`, plus chronology | **clean** — 0 incidents straddle; train ends 2024-12-29, test starts 2025-01-01 |
| `split_incident` | `group_incident` | **clean** — 0 straddle |
| `split_protocol` | `group_protocol` | **clean** — 0 straddle |
| `split_address` | `group_address` | **clean** — 0 straddle |
| `split_cross_chain` | — | metadata only: on an Ethereum-only transaction set its test side is empty by construction |

No duplicate hashes in any split. **No corrected split was needed**, so none was
published; had one been, it would have appeared under a new name
(`split_<name>_corrected`) alongside the original, never replacing it. The
upstream column is never modified.

**The temporal split is primary.** The protocol hold-out is reported separately.
The cross-chain split is kept as metadata.

### Validation partition

Carved out of **training rows only**, by whole incident groups and by time, so no
group straddles a boundary and validation stays earlier than test:

| partition | rows | attacks | negatives | time range |
|---|---|---|---|---|
| train | 586 | 227 | 359 | ≤ 2024-08-22 |
| validation | 147 | 30 | 117 | 2024-08-22 → 2024-12-29 |
| test | 899 | 287 | 612 | from 2025-01-01 |

`data/splits/primary_with_validation.parquet` is authoritative — `verify-splits`
builds it on top of a corrected assignment whenever one is needed, so training
always reads one file. Note the validation attack count is small (30): that is
the honest consequence of keeping incidents whole and validation in the past, and
it limits how finely a threshold can be tuned.

## Tasks

| task | positives | controls | ambiguous (left unresolved) | detectors |
|---|---|---|---|---|
| `exploit` *(main)* | 386 | 1,088 | — | BlockGPT, BlockScan, BlockLens, FT-Transformer |
| `key_compromise` | 158 | 1,088 | — | the same four |
| `price_manipulation` | 35 | 1,088 (frozen) | 40 | all five, DeFiGuard included |

Key compromises are excluded from the main task's positives, because a stolen key
leaves no contract-level weakness to detect, and are **reported separately rather
than merged into the main number**. BlockGPT, BlockScan, BlockLens and the
FT-Transformer can also be evaluated across general exploit families.

### The price-manipulation mapping

Reproducible and documented, over the release's own
`step3_oracle/output/attack_txs_accepted.csv` fields `vuln_type` and
`scwe_titles`. Patterns live in `PRICE_MANIPULATION_RULES`
(`src/txbench/tasks.py`) and every per-transaction decision, with the patterns it
matched, is written to `price_manipulation_decisions.parquet`.

- **included**: `vuln/oracle/price-manipulation`, `vuln/oracle/spot-price`,
  `vuln/oracle/stale-price`, `vuln/logic/price-calculation`,
  `Vulnerable Price Dependency`; or an SCWE title containing
  "Price Oracle Manipulation" or "Reliance on Low-Liquidity Spot Prices".
- **ambiguous, left unresolved**: `vuln/defi/sandwich-attack` and
  `vuln/defi/slippage` (ordering and missing guards, not a manipulated price
  read), `vuln/arithmetic/rounding`, and the SCWE titles
  "Missing Slippage Protection", "Extractable Value from Predictable Transaction
  Ordering", "ERC4626 Share Inflation via Donations", "Inconsistent Rounding
  Direction".
- **conflict rule**: a record matching any ambiguous pattern **stays ambiguous**
  even if it also matches an include pattern — the conflict *is* the ambiguity.
- Ambiguous records enter neither the positives nor the controls; all 40 are
  listed.
- Every one of the 544 attacks had a source record, so nothing was dropped for
  want of a mapping input.

The **legitimate control hashes are frozen** with the task: the hash list is
written to `task_price_manipulation.parquet` and its digest into `tasks.json`
(`control_keys_sha256`), so a later run cannot quietly redefine the controls.
`tests/test_real_release.py` asserts the identifiers line up across the five
eligible detectors.

## The prediction format

A CSV with `chain_id, tx_hash, model, version, split, status` required and
`score, score_direction, verdict, note` optional. Intake **refuses** rather than
warns:

- a duplicate row for one `(model, version, tx_key)`;
- a row claiming a **test hash was in training** ("test hashes must not appear in
  training");
- a declared split that disagrees with the frozen partition;
- `status: scored` with neither a score nor a binary verdict;
- an undeclared or unknown `score_direction` (`higher_is_attack` /
  `lower_is_attack`; scores are normalised so higher always means attack);
- a malformed hash or chain id, or a non-binary verdict.

Exactly **one prediction or one explicit failure per eligible hash** is required;
`status` may be `scored`, `failed`, `skipped` or `unsupported_input`. A hash with
no row at all is reported as missing and is **not** scored as a negative. Attack
labels never enter a detector's input at scoring time: this module reads the
predictions and the frozen manifest, and the two meet only here.

## Metrics

- The **operating threshold is selected on the validation partition only**
  (`max_f1` by default, or `target_fp_per_1k`), and the selection record states
  what it was fitted on and how many rows it saw. With no scored validation rows,
  no thresholded test metric is reported.
- Test results carry recall, precision, average precision, **realised** false
  positives per 1,000 labeled negatives, incident-weighted recall, and
  eligible/scored counts by class.
- A **binary-only detector** is reported at its actual operating point and gets
  no ranking metric, with that stated.
- A **paired comparison** runs on the hashes every model actually scored, next to
  each model's coverage of the full transaction set.
- Sampled negatives support historical comparison. **Alerts per day would require
  traffic observation and are not reported.** External pretraining and possible
  test overlap must be documented when a model is added.

## Parity with the pinned upstream evaluator

`bench evaluate` runs the release's own `step4_benign/evaluate.py` as a
subprocess, against an Ethereum-only copy of `combined.csv` written into a
working directory (the upstream file is never modified), and compares it with a
local replica on a fixed prediction fixture.

**Status: verified.** 16 shared keys compared — `rows`, `attacks`, `negatives`,
`incidents`, `without_prediction`, `tp`, `fp`, `fn`, `tn`, `precision`, `recall`,
`f1`, `false_positive_rate`, `recall_incidents_equal`, `split`, `track` — with
**zero differences** at tolerance 1e-9. That covers track membership (a track
keeps every negative and filters attacks by category), incident weighting (the
mean over incidents of the share of their attacks flagged) and the shared
metrics. The weight and negative-handling rules were read out of the upstream
script itself, not assumed. Without the evaluator, parity is marked
`unverified` instead.

One convention differs on purpose, and is recorded with the result: **upstream
counts a benchmark row with no prediction as not flagged**, while this project's
own metrics require an explicit row per eligible hash. On the fixture run the
upstream recall is 0.6124 over 129 attacks (2 unpredicted rows counted as
not-flagged) and the project recall is 0.6172 over the 128 attacks that carried a
prediction, with the failures reported separately. Project-specific metrics and
Ethereum-only results are identified as such.

## The fixture run

`fixtures/example_predictions.csv` is a deterministic pseudo-score over the 1,474
eligible hashes of the exploit task (1,471 scored, 3 explicit failures). It
exists to exercise intake, thresholding, metrics and parity end to end.

**It is not a detector, and no accuracy claim follows from it.** `--fixture-mode`
stamps that warning into the results file.
