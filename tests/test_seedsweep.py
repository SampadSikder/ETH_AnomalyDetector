"""A seed sweep is only evidence if every draw is kept and the seed took effect."""
from __future__ import annotations

import pytest

from txbench import seedsweep as S
from txbench.cli import _parse_seeds
from txbench.errors import EvaluationError, TxBenchError


def population(n_attacks=6, n_negatives=14):
    labels, incidents, classes = {}, {}, {}
    for i in range(n_attacks):
        key = f"1:0xa{i:03d}"
        labels[key], classes[key] = 1, "exploit"
        incidents[key] = f"inc{i // 2}"            # two transactions per incident
    for i in range(n_negatives):
        key = f"1:0xb{i:03d}"
        labels[key], incidents[key] = 0, key
    return labels, incidents, classes


def run_for(order, *, method="m", seed=42, effective=42, cutoffs=(5, 10)):
    """`order`: keys best-first. Scores descend so the ranking matches it."""
    labels, incidents, classes = population()
    scores = {k: float(len(order) - i) for i, k in enumerate(order)}
    return S.measure(scores, labels, method=method, seed=seed, effective_seed=effective,
                     keys=list(labels), cutoffs=cutoffs, incidents=incidents,
                     attack_classes=classes)


def test_measure_counts_attacks_at_each_cutoff():
    labels, _, _ = population()
    attacks = [k for k in labels if labels[k] == 1]
    negatives = [k for k in labels if labels[k] == 0]
    run = run_for(attacks + negatives)
    assert run.attacks_at(5) == 5
    assert run.attacks_at(10) == 6                  # only 6 attacks exist
    assert run.average_precision == pytest.approx(1.0)


def test_distinct_incidents_in_the_top_ten_is_counted_not_assumed():
    labels, _, _ = population()
    attacks = sorted(k for k in labels if labels[k] == 1)
    negatives = sorted(k for k in labels if labels[k] == 0)
    run = run_for(attacks + negatives)
    # Six attacks, two per incident: ten alerts tracing to three incidents.
    assert run.incidents_in_top10 == 3
    assert run.incident_names_in_top10 == ["inc0", "inc1", "inc2"]


def test_no_attacks_in_the_top_ten_reports_zero_incidents():
    labels, _, _ = population()
    negatives = sorted(k for k in labels if labels[k] == 0)
    attacks = sorted(k for k in labels if labels[k] == 1)
    run = run_for(negatives + attacks)
    assert run.attacks_at(10) == 0
    assert run.incidents_in_top10 == 0


def test_a_tie_crossing_a_cutoff_is_recorded_on_the_run():
    labels, incidents, classes = population()
    scores = {k: 1.0 for k in labels}               # everything tied
    run = S.measure(scores, labels, method="m", seed=42, effective_seed=42,
                    keys=list(labels), cutoffs=(5,), incidents=incidents,
                    attack_classes=classes)
    assert len(run.ties_crossing_cutoff) == 1
    tie = run.ties_crossing_cutoff[0]
    assert tie["k"] == 5 and tie["tied_at_cutoff"] == 20
    assert tie["reachable"] == [0, 5]
    assert run.at_k[5]["selection_determined"] is False


def test_reproduction_compares_counts_and_reports_a_mismatch():
    labels, _, _ = population()
    attacks = sorted(k for k in labels if labels[k] == 1)
    negatives = sorted(k for k in labels if labels[k] == 0)
    good = run_for(attacks + negatives)
    bad = run_for(negatives + attacks)
    assert S.reproduces(good, good, cutoffs=(5, 10))["reproduced"] is True
    check = S.reproduces(bad, good, cutoffs=(5, 10))
    assert check["reproduced"] is False
    assert check["checks"]["attacks_at_5"] == {"primary": 5, "rerun": 0, "match": False}


def test_summarise_keeps_every_seed_and_reports_the_spread():
    labels, _, _ = population()
    attacks = sorted(k for k in labels if labels[k] == 1)
    negatives = sorted(k for k in labels if labels[k] == 0)
    runs = [run_for(attacks + negatives, seed=42),
            run_for(negatives[:4] + attacks + negatives[4:], seed=43),
            run_for(negatives + attacks, seed=44)]
    block = S.summarise(runs, k=10)["m"]
    assert block["seeds"] == [42, 43, 44]
    assert block["n_seeds"] == 3                    # nothing dropped
    assert block["attacks_at_k"] == [6, 6, 0]
    assert block["min"] == 0 and block["max"] == 6 and block["range"] == 6
    assert block["mean"] == pytest.approx(4.0)


def test_head_to_head_is_a_per_seed_sign_count():
    labels, _, _ = population()
    attacks = sorted(k for k in labels if labels[k] == 1)
    negatives = sorted(k for k in labels if labels[k] == 0)
    strong, weak = attacks + negatives, negatives + attacks
    runs = [run_for(strong, method="challenger", seed=42),
            run_for(weak, method="baseline", seed=42),
            run_for(weak, method="challenger", seed=43),
            run_for(weak, method="baseline", seed=43),
            run_for(weak, method="challenger", seed=44),
            run_for(strong, method="baseline", seed=44)]
    head = S.head_to_head(runs, challenger="challenger", baseline="baseline", k=10)
    assert (head["wins"], head["ties"], head["losses"]) == (1, 1, 1)
    assert head["seeds_compared"] == 3
    assert head["win_rate"] == pytest.approx(1 / 3)
    assert head["margin_range"] == [-6, 6]
    assert [row["outcome"] for row in head["per_seed"]] == ["win", "tie", "loss"]


def test_head_to_head_refuses_an_unpaired_comparison():
    labels, _, _ = population()
    order = sorted(labels, key=lambda k: labels[k], reverse=True)
    runs = [run_for(order, method="a", seed=42)]
    with pytest.raises(EvaluationError):
        S.head_to_head(runs, challenger="a", baseline="b", k=10)


def test_the_table_lists_one_row_per_method_and_seed():
    labels, _, _ = population()
    attacks = sorted(k for k in labels if labels[k] == 1)
    negatives = sorted(k for k in labels if labels[k] == 0)
    runs = [run_for(attacks + negatives, method="a", seed=s) for s in (42, 43)]
    out = S.table(runs, [5, 10])
    assert out.count("\n") == 3                     # header, rule, two rows
    assert "42" in out and "43" in out


def test_seed_specs_parse_as_ranges_and_lists():
    assert _parse_seeds("42-51") == list(range(42, 52))
    assert _parse_seeds("42,44,43") == [42, 43, 44]
    assert _parse_seeds("42-44,50") == [42, 43, 44, 50]
    with pytest.raises(TxBenchError):
        _parse_seeds("51-42")
    with pytest.raises(TxBenchError):
        _parse_seeds("42,42")
    with pytest.raises(TxBenchError):
        _parse_seeds("")
