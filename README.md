# ETH_AnomalyDetector

Anomaly detection benchmark of SoTA detectors, compared against an FT-Transformer
classifier, on a frozen set of **real historical Ethereum mainnet transactions**.

Milestone 1 — this repository's current state — is the dataset and the evaluation
interface: a reproducible, labeled, auditable transaction set that a second researcher can
rebuild and check. **No model is trained yet, and no accuracy claim is made.**

## What exists today

| | |
|---|---|
| Transaction set | **1,632** Ethereum mainnet transactions: **544 attacks** (386 exploit, 158 key compromise) and **1,088** evidence-backed negatives |
| Source | [TxBench v1.0.1](https://doi.org/10.5281/zenodo.23758032), md5-verified, CC BY 4.0 ([paper](https://arxiv.org/abs/2611.04411v1)) |
| Transaction-list fingerprint | `b07e892a606422dff1fd280bcd940c60f400fcba67f6eb3231084c02f4d81fcf` (sorted `tx_key`, `label`) |
| Quarantined | 0 rows — and an unknown label, an RPC failure or a revert is never read as benign |
| RPC calls needed | **0** for the base dataset; the release already supplies every envelope, receipt and call tree |
| Feature views | two, both at 100 % coverage: `tx-receipt` (690 tokens) and `full-calls-receipt` (2,096 tokens) |
| Splits | all five upstream splits preserved and audited clean; validation carved out of training (586 / 147 / 899) |
| Detector inputs | FT-Transformer 1,632 usable; DeFiGuard 1,590 partial / 42 blocked; BlockScan, BlockGPT, BlockLens blocked, each with a named missing field |
| Upstream evaluator | parity **verified** against the release's own `evaluate.py` across 16 metrics, zero differences |
| Tests | 125 offline tests, no credentials required |

The headline finding: **the base dataset costs nothing to rebuild.** Because the
release ships enriched traces, the collector's inventory step reduces the first
pass from the budgeted 3,264 calls / 65,280 CU to zero. That budget is still
priced, as the cross-check scenario, in `reports/status.md`.

The binding limitation: the release's `internal_transactions` is an **ordered flat
frame list with no trace address, parent pointer or depth**. Call nesting is
therefore unavailable, which is what blocks BlockScan and BlockGPT. See
[docs/DETECTORS.md](docs/DETECTORS.md) for the exact missing profile per detector
and the cheapest enrichment that would unlock each one.

## Environment

Every script, test and pipeline command runs in a dedicated, non-base Miniconda
environment. System Python and Conda's `base` are not supported.

```sh
conda env create -n txbench -f environment.yml
conda run -n txbench python -c "import sys; print(sys.executable)"
# -> .../envs/txbench/bin/python
```

Credentials are optional and live in a git-ignored `.env` at the repository root
(`cp .env.example .env`). `ALCHEMY_API_KEY` is the primary provider,
`DRPC_API_KEY` an optional fallback; with only the Alchemy key present the
collector runs on Alchemy alone. Offline imports, dry runs and the fixture tests
work with neither key. Keys never reach a log, report, cache record or error
message — see [src/txbench/credentials.py](src/txbench/credentials.py) and
`tests/test_security.py`.

## Rebuild

```sh
conda run -n txbench bench fetch-release                      # optional: pull + verify the release
conda run -n txbench bench import-txbench --release-dir data/upstream/txbench --chain-id 1
# optional, for BlockScan's preprocessing smoke test:
git clone https://github.com/nuwuxian/BlockScan.git data/upstream/detectors/blockscan
git -C data/upstream/detectors/blockscan checkout 4556cd209c968ed60e8d15b245f8d53e3f725534
conda run -n txbench bench collect --dry-run
conda run -n txbench bench collect --pilot 20 --seed 42 --provider alchemy --fallback drpc
conda run -n txbench bench audit-inputs --pilot
conda run -n txbench bench collect --resume --provider alchemy --fallback drpc
conda run -n txbench bench verify-splits
conda run -n txbench bench build-features --view tx-receipt
conda run -n txbench bench build-features --view full-calls-receipt
conda run -n txbench bench build-tasks
conda run -n txbench bench validate-adapters
conda run -n txbench bench evaluate --predictions fixtures/example_predictions.csv --fixture-mode
conda run -n txbench bench report
```

`conda run -n txbench bench rebuild` prints that sequence. Tests:
`conda run -n txbench python -m pytest`.

## Documentation

- [docs/DATASET.md](docs/DATASET.md) — the transaction set, its provenance, the schema mapping, licensing
- [docs/COLLECTION.md](docs/COLLECTION.md) — the cache-first collector, budgets, providers, enrichment routes
- [docs/FEATURES.md](docs/FEATURES.md) — the two feature views and their exact definitions
- [docs/DETECTORS.md](docs/DETECTORS.md) — per-detector field contracts, evidence levels, coverage
- [docs/EVALUATION.md](docs/EVALUATION.md) — tasks, splits, metrics, the prediction format, upstream parity
- `reports/status.md` — the generated status report, with every artifact path

## Layout

```
src/txbench/        the package: importer, collector, features, adapters, evaluation
config/             benchmark.toml - overridable limits, logged with every run
fixtures/           a marked FIXTURE release and an example prediction file
tests/              offline tests plus real-data acceptance checks
data/   (ignored)   upstream release, raw cache, manifest, features, splits, evaluation
reports/ (ignored)  generated transaction-set and status reports
```

Upstream data is treated as read-only evidence: it is never modified, never mixed
with model features, and every file read is recorded with its checksum. This
repository redistributes none of it — see "Licensing" in
[docs/DATASET.md](docs/DATASET.md).
