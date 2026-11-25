# Detector input validation

Per-transaction compatibility rows live in
`data/adapters/cohort/compatibility_table.parquet` (detector, verdict, required
fields available, required fields missing, missing-field reason, notes), and the
summary in `data/adapters/cohort/adapter_report.json`.

## Evidence levels, kept strictly apart

Conflating these is how a benchmark starts reporting an input as usable when
nobody ran anything, so each is a distinct value:

| level | meaning |
|---|---|
| `schema_check` | a field contract was checked against this project's data |
| `upstream_preprocess_ok` | a converted record ran through the detector's **own** preprocessing entry point, at full fidelity |
| `upstream_preprocess_degraded` | that entry point ran, but on a record missing a required field class — a documented deviation, **never** reported as a pass |
| `paper_reimplementation` | this project's adapter, written from the paper |
| `project_implementation` | a component this project owns outright |

Per-transaction verdicts are `usable`, `partial` or `blocked`, and **a blocked
record stays in the denominator.**

## Coverage on the real cohort (1,632 transactions)

| detector | eligible | usable | partial | blocked | binding missing field | evidence attained |
|---|---|---|---|---|---|---|
| **FT-Transformer** | 1,632 | **1,632** | 0 | 0 | — | `project_implementation` |
| **DeFiGuard** | 1,632 | 0 | **1,590** | 42 | `node_type` (all), `native_flows` (42) | `paper_reimplementation` |
| **BlockScan** | 1,632 | 0 | 0 | **1,632** | `nested_calls` | `schema_check`, `upstream_preprocess_degraded` |
| **BlockGPT** | 1,632 | 0 | 0 | **1,632** | `ordered_state_access`, `nested_structure` | `schema_check` |
| **BlockLens** | 1,632 | 0 | 0 | **1,632** | `opcode_sequence`, `step_operands` | `schema_check` |

## FT-Transformer

Pinned to this project's own `txbench.features.build_sparse_view`; the sample
schema is each view's `feature_dictionary.json`. Both documented views and the
labels are exported with equal train/test column definitions within each view,
and each view's coverage is reported separately (both 100 %).

## BlockScan

[Authors' repository](https://github.com/nuwuxian/BlockScan), pinned at
**`4556cd209c968ed60e8d15b245f8d53e3f725534`**. The checkout lives at
`data/upstream/detectors/blockscan` (git-ignored, like all upstream material):

```sh
git clone https://github.com/nuwuxian/BlockScan.git data/upstream/detectors/blockscan
git -C data/upstream/detectors/blockscan checkout 4556cd209c968ed60e8d15b245f8d53e3f725534
```

Entry point:
`src_eth/preprocess.py::extract_function_calls`, driven by `write_all_traces`.

The field contract was read off that file rather than guessed. It expects, per
transaction, `{"<tx_hash>": [record]}` where the record is **recursively
nested**:

```
{type, from, to, func, gas, value,
 args:   [{type, data}],        # typed, decoded arguments
 output: [{type, data}],
 calls:  [ …the same shape, recursively… ],
 logs:   [{address, topics: [{type, data}], data: [{type, data}]}]}
```

Findings, all recorded rather than worked around:

- **`calls` nesting is unavailable** for every transaction: the release supplies
  an ordered flat frame list with no trace address, parent pointer or depth. This
  is what blocks the detector.
- **Frame-level decoded arguments are unavailable**: the release decodes the
  outer call only (`decoded_call`); internal frames carry raw input/output bytes
  with no ABI types.
- **Typed log topics are incomplete for 153 transactions**: 18 of them carry logs
  with **no topic 0** at all (16 with one such log, 2 with two) — and the pinned
  preprocessor indexes `log['topics'][0]` unconditionally, so it would raise on
  them. The rest lack a decoded, typed form upstream for some of their logs
  (often because no verified ABI exists in Sourcify for the target contract).
- **The driver cannot run as shipped.** `write_all_traces` reads
  `config/top_20k_address.txt` and `config/top_20k_hash.txt`; the pinned revision
  contains **no `config/` and no `data/` directory** — those are distributed only
  through an external drive link. There is therefore also no upstream example
  record in the repository to test against.

What was actually run: the pinned `extract_function_calls` is imported by path
and executed on a converted pilot record. It runs and produces a token sequence
(134 tokens for the reconciled transaction), and because our converted record
carries no nested `calls`, the result is recorded as
**`upstream_preprocess_degraded`** with its reason — not as a pass. Zero-topic
logs are omitted from the conversion and counted, rather than being fed to code
that would crash on them. See `tests/test_adapters.py`.

## BlockGPT

[Blockchain Large Language Models](https://arxiv.org/abs/2304.12749). No authors'
preprocessing entry point is pinned here, so any adapter would be a paper-based
reimplementation and is labelled as such; none is claimed.

Its published representation needs the call tree with per-frame inputs and
outputs, the log records, **and the ordered storage reads and writes with their
keys and values**. The call frames and logs are available; the tree *shape* is
not (same nesting gap as BlockScan), and no state access is available at all.
A pre/post state difference cannot reconstruct every intermediate read or write,
so no substitute is offered. Both gaps are flagged per transaction.

## BlockLens

[IACR ePrint 2025/1634](https://eprint.iacr.org/2025/1634), Appendix A (the trace
tokenizer). It needs the ordered opcode sequence with operands and the
stack/memory/storage information the tokenizer reads. None exists for this
cohort. Transaction boundaries and a deterministic chunk order are available and
would be derivable once an instruction stream existed.

Top-level calldata and nested call trees cannot supply an instruction sequence,
so **no amount of call-level enrichment unblocks this detector** — it needs
struct-log tracing.

## DeFiGuard

[DeFiGuard](https://arxiv.org/abs/2406.11157). Reimplemented from the paper in
`src/txbench/flows.py` and labelled `paper_reimplementation`. Directed
native-asset and ERC-20 flow edges are derived from call values and transfer
logs, with parties, asset and amount.

The paper's four node features (Table I) are pinned: **node type** (account kind
*and* source-code transparency), **transfer frequency**, **transfer diversity**,
**profit score**. Three are computed from the edges. **Node type is not
defaulted**: it needs each address's account kind (one `eth_getCode` per distinct
address) and its verification status (an external explorer index), so it is
reported missing.

Ambiguities, each named rather than guessed:

- **Call types that move no value**: DELEGATECALL and STATICCALL cannot transfer
  the callee's balance, so their `value` field never becomes an edge; such frames
  are counted as skipped.
- **Reverted ancestry**: a frame undone by an *ancestor's* failure leaves no state
  change, but a flat frame list cannot say who an ancestor is. Native edges are
  emitted only from a transaction whose receipt succeeded and whose frames report
  no error; otherwise the native side is marked incomplete. This is what blocks
  **42** transactions (41 with failed frames, 1 whose status is absent because the
  receipt is pre-Byzantium).
- **ERC-20 / ERC-721 signature ambiguity**: both use one `Transfer` topic 0. The
  standard shapes separate (three topics + 32 data bytes → fungible; four topics
  + empty data → a token id, which has no fungible amount); anything else stays
  `ambiguous`. 47 such edges exist in the cohort.
- **A half-known transfer is excluded from the profit score.** Debiting a sender
  whose counterparty could not be resolved would invent a loss, so the edge is
  kept and the exclusion is counted with its reason.

Coverage outcomes are reported as three separate results, as required:
**1,495 transfers observed**, **95 genuine no-transfer transactions**, **42
incomplete graphs**.

## The cheapest next enrichment

| what it would unlock | field | transactions | source |
|---|---|---|---|
| **BlockScan** (and BlockGPT's tree shape) | `nested_calls` / `nested_structure` | 1,632 | a call-tracer export (`debug_traceTransaction` callTracer), a Parity-style trace with `trace_address`, or the public BigQuery Ethereum traces table |
| DeFiGuard's fourth node feature, and `reverted_ancestry` for the 42 blocked graphs | `node_type`, `reverted_ancestry` | 1,632 / 42 | one `eth_getCode` per distinct address (within the free plan) plus an external source-verification index; nesting for ancestry |
| **BlockGPT** fully | `ordered_state_access` | 1,632 | struct-log or prestate tracing on an archive node |
| **BlockLens** | `opcode_sequence`, `step_operands` | 1,632 | struct-log tracing on an archive node |

**Call nesting is the single cheapest unlock**: it is the only missing field that
a free, already-public source could plausibly supply, and it unblocks BlockScan
while also giving BlockGPT its tree shape and DeFiGuard its revert ancestry.
Opcode-level data is the expensive one and is needed by two of the five.

The exact hashes needing each field are listed in the `enrichment_plan` block of
`data/adapters/cohort/adapter_report.json`.
