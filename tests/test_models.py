"""The three detector runs: inputs excluded, failures stated, metrics per class."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from conftest import H, REPO

from txbench import evaluate as E
from txbench.flows import FlowEdge, FlowGraph, NATIVE_ASSET
from txbench.models import PredictionRow, write_predictions_csv
from txbench.models import defiguard_gnn as G

torch = pytest.importorskip("torch", reason="the model runs need torch")


def _graph(tx_key: str, *, complete: bool = True, edges: int = 2) -> FlowGraph:
    graph = FlowGraph(tx_key=tx_key, native_complete=complete, token_complete=True)
    if not complete:
        graph.native_reason = "receipt status unavailable (absent_pre_byzantium)"
    for i in range(edges):
        graph.edges.append(FlowEdge(sender=f"0x{i:040x}", receiver=f"0x{i + 1:040x}",
                                    asset=NATIVE_ASSET, asset_kind="native",
                                    amount=10 ** 18 * (i + 1), source="call_frame", reference=i))
    return graph


def test_prediction_rows_pass_the_evaluator_intake(tmp_path):
    rows = [PredictionRow(tx_key=f"1:{H(1)}", model="m", version="v", split="test", score=0.75),
            PredictionRow(tx_key=f"1:{H(2)}", model="m", version="v", split="test",
                          status="unsupported_input", note="flow graph incomplete")]
    path = write_predictions_csv(rows, tmp_path / "p.csv")
    bundles = E.read_predictions(path)
    bundle = bundles["m@v"]
    assert bundle.predictions[f"1:{H(1)}"].score == 0.75
    assert bundle.predictions[f"1:{H(2)}"].status == "unsupported_input"
    assert bundle.binary_only is False, "a continuous score must not be read as a verdict"


def test_a_score_is_never_invented_for_an_unscorable_row(tmp_path):
    path = write_predictions_csv(
        [PredictionRow(tx_key=f"1:{H(3)}", model="m", version="v", split="train",
                       status="unsupported_input", note="no graph")], tmp_path / "p.csv")
    with open(path, encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["score"] == "", "a failure row must carry no number at all"
    assert row["verdict"] == ""


# --- DeFiGuard ----------------------------------------------------------------

def test_node_type_is_absent_from_the_input_dimension():
    assert "node_type" not in G.NODE_FEATURES
    assert G.OMITTED_NODE_FEATURES == ("node_type",)
    graph = G.to_tx_graph(f"1:{H(1)}", _graph(f"1:{H(1)}"))
    assert graph.x.shape[1] == len(G.NODE_FEATURES) == 5


def test_an_incomplete_graph_is_excluded_with_its_reason(monkeypatch):
    keys = [f"1:{H(1)}", f"1:{H(2)}"]
    graphs = {keys[0]: _graph(keys[0]), keys[1]: _graph(keys[1], complete=False)}

    class FakeTx:
        pass

    monkeypatch.setattr(G, "build_flow_graph", lambda tx: graphs[tx.tx_key])
    records = []
    for key in keys:
        tx = FakeTx()
        tx.tx_key = key
        records.append((key, tx))
    built, excluded = G.build_graphs(records, {k: "train" for k in keys})
    assert [g.tx_key for g in built] == [keys[0]]
    assert len(excluded) == 1
    assert excluded[0].tx_key == keys[1]
    assert "pre_byzantium" in excluded[0].reason
    assert excluded[0].outcome == "incomplete_graph"


def test_a_transaction_with_no_transfers_is_scored_with_an_explicit_flag():
    empty = FlowGraph(tx_key=f"1:{H(4)}", native_complete=True, token_complete=True)
    assert empty.outcome == "no_transfers_observed"
    graph = G.to_tx_graph(empty.tx_key, empty)
    assert graph.n_nodes == 0 and graph.n_edges == 0
    assert graph.graph_x[G.GRAPH_FEATURES.index("no_transfers_observed")] == 1.0
    model = G.DeFiGuardGNN(seed=1).eval()
    with torch.no_grad():
        value = float(model.forward(G.collate([graph]))[0])
    assert np.isfinite(value), "an empty graph must still produce a finite score"


def test_a_forward_pass_on_a_real_shaped_graph_is_finite_and_batches_match():
    graphs = [G.to_tx_graph(f"1:{H(i)}", _graph(f"1:{H(i)}", edges=i + 1)) for i in (1, 2, 3)]
    model = G.DeFiGuardGNN(seed=7).eval()
    with torch.no_grad():
        batched = model.forward(G.collate(graphs))
        singles = [float(model.forward(G.collate([g]))[0]) for g in graphs]
    assert batched.shape == (3,)
    assert all(np.isfinite(v) for v in singles)
    # a batch must not let one graph's nodes leak into another's readout
    for position, value in enumerate(singles):
        assert abs(float(batched[position]) - value) < 1e-5


def test_the_scaler_is_fitted_on_training_graphs_only():
    train = [G.to_tx_graph(f"1:{H(1)}", _graph(f"1:{H(1)}", edges=1))]
    stats = G.standardize_fit(train)
    assert stats["fitted_on"] == "training rows only"
    assert len(stats["node_mean"]) == len(G.NODE_FEATURES)
    assert all(s > 0 for s in stats["node_std"])


# --- FT-Transformer -----------------------------------------------------------

def test_the_feature_view_keeps_identifiers_and_labels_out_of_the_inputs():
    from txbench.models import ft_transformer as F
    features = REPO / "data" / "features"
    if not (features / "full_calls_receipt" / "vocab_temporal.json").is_file():
        pytest.skip("no built feature view in this checkout")
    data = F.load_view(features, "full-calls-receipt", min_document_frequency=5)
    for forbidden in ("tx_key", "partition", "label", "usable", "unusable_reason",
                      "group_incident", "attack_class", "block_timestamp"):
        assert forbidden not in data.columns
    assert data.scaler["fitted_on"] == "training rows only"
    assert data.selection["document_frequency_fitted_on"] == "training rows only"


def test_the_transformer_reads_only_the_cls_token_and_is_deterministic():
    from txbench.models import ft_transformer as F
    model = F.FTTransformer(6, d_token=8, n_blocks=1, n_heads=2, seed=3).eval()
    x = np.arange(12, dtype=np.float32).reshape(2, 6)
    first = F.predict(model, x)
    second = F.predict(model, x)
    assert len(first) == 2 and first == second
    assert all(0.0 <= v <= 1.0 for v in first)


# --- the evaluator ------------------------------------------------------------

def test_recall_is_reported_per_attack_class_with_its_denominators(tmp_path):
    keys = [f"1:{H(i)}" for i in range(1, 6)]
    labels = {keys[0]: 1, keys[1]: 1, keys[2]: 1, keys[3]: 0, keys[4]: 0}
    classes = {keys[0]: "exploit", keys[1]: "exploit", keys[2]: "key_compromise"}
    rows = [PredictionRow(tx_key=keys[0], model="m", version="v", split="test", score=0.9),
            PredictionRow(tx_key=keys[1], model="m", version="v", split="test", score=0.1),
            PredictionRow(tx_key=keys[2], model="m", version="v", split="test",
                          status="failed", note="no score"),
            PredictionRow(tx_key=keys[3], model="m", version="v", split="test", score=0.8),
            PredictionRow(tx_key=keys[4], model="m", version="v", split="test", score=0.2)]
    bundles = E.read_predictions(write_predictions_csv(rows, tmp_path / "p.csv"))
    result = E.score_partition(bundles["m@v"], labels, keys, threshold=0.5,
                               attack_classes=classes, partition="test")
    per_class = result["recall_by_attack_class"]
    assert per_class["exploit"] == {"eligible": 2, "scored": 2, "detected": 1,
                                    "recall": 0.5, "recall_over_eligible": 0.5}
    # the unscored key compromise is not silently counted as a miss in `recall`
    assert per_class["key_compromise"]["scored"] == 0
    assert per_class["key_compromise"]["recall"] is None
    assert per_class["key_compromise"]["recall_over_eligible"] == 0.0
    assert "negative" not in per_class, "a negative has no attack class"
    assert result["false_positives"] == 1 and result["negatives_scored"] == 2


def test_a_frozen_threshold_is_applied_verbatim(tmp_path):
    keys = [f"1:{H(1)}", f"1:{H(2)}"]
    rows = [PredictionRow(tx_key=keys[0], model="m", version="v", split="test", score=0.6),
            PredictionRow(tx_key=keys[1], model="m", version="v", split="test", score=0.4)]
    bundles = E.read_predictions(write_predictions_csv(rows, tmp_path / "p.csv"))
    labels = {keys[0]: 1, keys[1]: 0}
    strict = E.score_partition(bundles["m@v"], labels, keys, threshold=0.55)
    loose = E.score_partition(bundles["m@v"], labels, keys, threshold=0.35)
    assert (strict["tp"], strict["fp"]) == (1, 0)
    assert (loose["tp"], loose["fp"]) == (1, 1)


# --- BlockScan assets ---------------------------------------------------------

def test_a_missing_blockscan_asset_is_named_with_where_it_comes_from(tmp_path):
    from txbench.models import blockscan_run as B
    report = B.asset_report(tmp_path)
    assert report["complete"] is False
    assets = {m["asset"] for m in report["missing"]}
    assert assets == {"vocabulary_addresses", "vocabulary_selectors", "tokenizer", "checkpoint"}
    for item in report["missing"]:
        assert item["source"], "a missing asset must say where it comes from"
        assert item["needed_by"]


def test_the_device_shims_are_declared_and_the_xformers_stub_refuses_use():
    from txbench.models import blockscan_score as S
    assert len(S.SHIMS) == 3
    assert all(shim["reason"] for shim in S.SHIMS)
    assert any("xformers" in shim["target"] for shim in S.SHIMS)


def test_a_tx_path_maps_back_to_its_key():
    from txbench.models import blockscan_run as B
    assert B.key_of_path(f"/work/test/{H(1)}.json\n", 1) == f"1:{H(1)}"


def test_staging_writes_the_authors_shape_and_maps_back(tmp_path, tx_of):
    """The staged file is `{tx_hash: [record]}`, keyed so tx_path.txt maps back."""
    from txbench.adapters.blockscan_smoke import to_blockscan_record
    from txbench.models import blockscan_run as B

    tx = tx_of(1)
    staged, failures = B.stage_records([(tx.tx_key, tx)], {tx.tx_key: "train"}, tmp_path,
                                       extract_function_calls=lambda record: list(range(7)))
    assert failures == []
    slot = staged["train"]
    written = list(slot.directory.glob("*.json"))
    assert len(written) == 1
    payload = json.loads(written[0].read_text())
    tx_hash = tx.tx_key.split(":", 1)[1]
    assert list(payload) == [tx_hash]
    assert payload[tx_hash][0] == to_blockscan_record(tx)
    assert B.key_of_path(str(written[0]), 1) == tx.tx_key
    assert slot.raw_token_counts[tx.tx_key] == 7


def test_a_record_without_a_partition_is_a_failure_not_a_silent_drop(tmp_path, tx_of):
    from txbench.models import blockscan_run as B
    tx = tx_of(1)
    staged, failures = B.stage_records([(tx.tx_key, tx), ("1:0xdeadbeef", None)],
                                       {tx.tx_key: "test"}, tmp_path,
                                       extract_function_calls=lambda record: [])
    assert failures == ["1:0xdeadbeef"]
    assert set(staged) == {"test"}


def test_the_truncation_summary_counts_what_the_model_never_saw():
    from txbench.models import blockscan_run as B
    rows = [PredictionRow(tx_key=f"1:{H(1)}", model="blockscan", version="v", split="test",
                          score=3.0,
                          extra={"tokens_before_truncation": 5000, "tokens_scored": 1024,
                                 "masked_tokens": 153, "truncated": True,
                                 "trace_tokens_before_oov_removal": 900,
                                 "trace_tokens_after_oov_removal": 700,
                                 "oov_tokens_removed": 200}),
            PredictionRow(tx_key=f"1:{H(2)}", model="blockscan", version="v", split="test",
                          score=1.0,
                          extra={"tokens_before_truncation": 40, "tokens_scored": 40,
                                 "masked_tokens": 6, "truncated": False,
                                 "trace_tokens_before_oov_removal": 100,
                                 "trace_tokens_after_oov_removal": 100,
                                 "oov_tokens_removed": 0})]
    summary = B.truncation_summary(rows)
    assert summary["truncated_transactions"] == 1
    assert summary["truncated_share"] == 0.5
    assert summary["model_tokens_before_truncation"]["max"] == 5000
    assert summary["oov_tokens_removed"]["total"] == 200
    assert summary["oov_removal_share_of_trace"] == 0.2
    assert any("truncation=True" in note for note in summary["notes"])
