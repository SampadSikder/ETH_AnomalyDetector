"""Feature views: train-only fitting, OOV kept, missing never zero."""
from __future__ import annotations

import pandas as pd

from conftest import H

from txbench import features as F


def _sparse(records, view=F.VIEW_TX_RECEIPT):
    return F.build_sparse_view(view, sorted(records.items()))


def test_vocabulary_is_fitted_on_training_rows_only(records, manifest):
    sparse = _sparse(records)
    train = [row.tx_key for row in manifest.itertuples(index=False)
             if row.split_temporal == "train"]
    test = [row.tx_key for row in manifest.itertuples(index=False)
            if row.split_temporal == "test"]
    vocab = F.fit_vocabulary(sparse, train, split="temporal")
    by_key = {r.tx_key: r for r in sparse.rows}
    test_only = set()
    for key in test:
        test_only |= set(by_key[key].counts)
    for key in train:
        test_only -= set(by_key[key].counts)
    assert test_only, "the fixture must contain a token only the test rows have"
    assert not (test_only & set(vocab.tokens)), "a test-only token must not enter the vocabulary"
    assert vocab.fitted_on_rows == len([k for k in train if by_key[k].status.get("usable")])


def test_out_of_vocabulary_mass_is_counted_not_dropped(records, manifest):
    sparse = _sparse(records)
    train = [row.tx_key for row in manifest.itertuples(index=False)
             if row.split_temporal == "train"]
    test = [row.tx_key for row in manifest.itertuples(index=False)
            if row.split_temporal == "test"]
    vocab = F.fit_vocabulary(sparse, train, split="temporal")
    labels = {row.tx_key: int(row.label) for row in manifest.itertuples(index=False)}
    matrix = F.build_matrix(sparse, vocab, test, labels, "test")
    oov = matrix[list(F.OOV_COLUMNS)].fillna(0).to_numpy().sum()
    assert oov > 0, "the fixture's test rows carry tokens outside the training vocabulary"


def test_a_row_missing_from_a_view_is_null_not_zero(records, manifest):
    sparse = _sparse(records, F.VIEW_FULL_CALLS)
    labels = {row.tx_key: int(row.label) for row in manifest.itertuples(index=False)}
    vocab = F.fit_vocabulary(sparse, [r.tx_key for r in sparse.rows], split="temporal")
    partial_key = f"1:{H(4)}"        # upstream marks its call tree 'partial'
    matrix = F.build_matrix(sparse, vocab, [partial_key], labels, "test")
    row = matrix.iloc[0]
    assert row["usable"] is False or row["usable"] == False      # noqa: E712
    assert row["unusable_reason"]
    feature_columns = [c for c in matrix.columns if c.startswith(("sel:", "tpc0:", "pre:"))]
    assert all(pd.isna(row[c]) for c in feature_columns), \
        "an unavailable row must be null across every feature, never zero"


def test_an_observed_absence_is_a_zero(records, manifest):
    """A receipt that exists with no logs gives genuine zero topic counts."""
    sparse = _sparse(records)
    labels = {row.tx_key: int(row.label) for row in manifest.itertuples(index=False)}
    keys = [r.tx_key for r in sparse.rows]
    vocab = F.fit_vocabulary(sparse, keys, split="temporal")
    empty_logs_key = f"1:{H(2)}"
    matrix = F.build_matrix(sparse, vocab, [empty_logs_key], labels, "train")
    row = matrix.iloc[0]
    assert bool(row["usable"]) is True
    topic_columns = [c for c in matrix.columns if c.startswith("tpc0:")]
    assert topic_columns, "the fixture vocabulary has topic tokens"
    assert all(row[c] == 0 for c in topic_columns)


def test_model_features_exclude_identifiers_and_label_evidence(records, manifest):
    sparse = _sparse(records)
    labels = {row.tx_key: int(row.label) for row in manifest.itertuples(index=False)}
    keys = [r.tx_key for r in sparse.rows]
    vocab = F.fit_vocabulary(sparse, keys, split="temporal")
    matrix = F.build_matrix(sparse, vocab, keys, labels, "train")
    forbidden = {"attack_class", "label_tier", "source_category", "group_incident",
                 "behavior_tags", "release_id", "upstream_trace_path", "subcategory",
                 "is_key_compromise", "provenance_sha256"}
    assert not (forbidden & set(matrix.columns))
    dictionary = F.feature_dictionary(sparse.view, vocab)
    assert dictionary["target_column"] == "label"
    assert dictionary["key_column"] == "tx_key"
    for name in forbidden:
        assert name in dictionary["non_feature_columns"] or name not in matrix.columns


def test_scalers_are_fitted_on_training_rows_only(records, manifest):
    sparse = _sparse(records)
    labels = {row.tx_key: int(row.label) for row in manifest.itertuples(index=False)}
    train = [row.tx_key for row in manifest.itertuples(index=False)
             if row.split_temporal == "train"]
    vocab = F.fit_vocabulary(sparse, train, split="temporal")
    matrix = F.build_matrix(sparse, vocab, train, labels, "train")
    scalers = F.fit_scalers(matrix, vocab)
    assert scalers["fitted_on"] == "training rows only"
    assert scalers["fitted_on_rows"] == int(matrix["usable"].sum())
    assert set(scalers["columns"]) == set(vocab.tokens) | set(F.OOV_COLUMNS)


def test_unusable_rows_stay_in_the_denominator(records, manifest):
    sparse = _sparse(records, F.VIEW_FULL_CALLS)
    labels = {row.tx_key: int(row.label) for row in manifest.itertuples(index=False)}
    report = F.coverage(sparse, labels)
    assert report["eligible_rows"] == len(sparse.rows)
    assert report["usable_rows"] < report["eligible_rows"], "the fixture has a partial trace"
    assert report["unusable_by_reason"]
    assert sum(c["eligible"] for c in report["by_class"].values()) == report["eligible_rows"]


def test_sparse_counts_are_written_before_any_vocabulary(records):
    sparse = _sparse(records)
    long = sparse.to_long_frame()
    assert set(long.columns) == {"tx_key", "token", "count", "namespace"}
    assert set(long["namespace"]) <= {"sel", "tpc0", "pre"}
