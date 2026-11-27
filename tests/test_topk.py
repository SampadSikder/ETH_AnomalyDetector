"""Top-k is only honest if the tie-break is label-blind and ties are declared."""
from __future__ import annotations

import pytest

from txbench import topk as T
from txbench.errors import EvaluationError


def build(scored):
    """scored: {hash_suffix: (score, label)} -> a ranking."""
    scores = {f"1:0x{h}": s for h, (s, _) in scored.items()}
    labels = {f"1:0x{h}": l for h, (_, l) in scored.items()}
    return T.rank(scores, labels), scores, labels


def test_ties_break_on_ascending_hash_not_on_the_label():
    # Every score identical, so hash order alone decides. The attack has the
    # later hash: a label-aware tie-break would pull it to the front.
    ordered, _, _ = build({"aa": (1.0, 0), "bb": (1.0, 0), "cc": (1.0, 1)})
    assert [r.tx_hash for r in ordered] == ["0xaa", "0xbb", "0xcc"]
    assert T.top_k(ordered, 1, method="m").attacks_found == 0


def test_higher_scores_come_first():
    ordered, _, _ = build({"aa": (0.1, 0), "bb": (0.9, 1), "cc": (0.5, 0)})
    assert [r.tx_hash for r in ordered] == ["0xbb", "0xcc", "0xaa"]
    assert [r.rank for r in ordered] == [1, 2, 3]


def test_attacks_plus_false_alerts_always_equals_k():
    ordered, _, _ = build({f"{i:02x}": (float(i), i % 3 == 0) for i in range(20)})
    for k in (1, 5, 10, 15, 20):
        r = T.top_k(ordered, k, method="m")
        assert r.attacks_found + r.false_alerts == k
        assert r.precision_at_k == pytest.approx(r.attacks_found / k)


def test_a_tie_straddling_the_cutoff_is_reported_as_a_range():
    # Ranks 1-2 are fixed; ranks 3-5 tie, and k=4 takes two of those three.
    ordered, _, _ = build({"aa": (9.0, 1), "bb": (8.0, 0),
                           "cc": (5.0, 1), "dd": (5.0, 1), "ee": (5.0, 0)})
    r = T.top_k(ordered, 4, method="m")
    assert r.tie_at_cutoff is True
    assert r.attacks_found == 3                      # cc and dd by hash order
    assert (r.attacks_found_min, r.attacks_found_max) == (2, 3)
    assert r.attacks_found_min <= r.attacks_found <= r.attacks_found_max
    assert r.tied_block["tied_at_cutoff"] == 3
    assert r.tied_block["slots_remaining"] == 2
    # Two of three tied, two of which are attacks: 2 * 2/3.
    assert r.expected_attacks_under_random_tie_order == pytest.approx(1 + 4 / 3)


def test_a_cutoff_that_lands_cleanly_reports_no_range():
    ordered, _, _ = build({"aa": (9.0, 1), "bb": (8.0, 0), "cc": (5.0, 1)})
    r = T.top_k(ordered, 2, method="m")
    assert r.tie_at_cutoff is False
    assert r.attacks_found_min == r.attacks_found_max == r.attacks_found
    assert r.tied_block is None
    assert r.selection_determined is True


def test_a_selection_wholly_inside_one_tie_is_flagged_as_undetermined():
    # The saturating-length case: one score covers everything at the top, so no
    # transaction is fixed above the cutoff and the tie-break picks the lot.
    ordered, _, _ = build({f"{i:02x}": (153.0, i < 4) for i in range(10)})
    r = T.top_k(ordered, 5, method="length")
    assert r.selection_determined is False
    assert (r.attacks_found_min, r.attacks_found_max) == (0, 4)
    assert r.expected_attacks_under_random_tie_order == pytest.approx(5 * 4 / 10)
    assert "warning" in r.to_dict()
    assert "*" in T.table([r]) and "hash tie-break" in T.table([r])


def test_recall_and_fpr_use_the_partition_counts_as_denominators():
    ordered, _, _ = build({"aa": (9.0, 1), "bb": (8.0, 1), "cc": (7.0, 0),
                           "dd": (6.0, 0), "ee": (5.0, 0)})
    r = T.top_k(ordered, 3, method="m")
    assert r.attacks == 2 and r.negatives == 3
    assert r.recall_at_k == pytest.approx(2 / 2)
    assert r.false_positive_rate == pytest.approx(1 / 3)


def test_ranking_refuses_an_unscored_hash_instead_of_dropping_it():
    scores = {"1:0xaa": 1.0}
    labels = {"1:0xaa": 1, "1:0xbb": 0}
    with pytest.raises(EvaluationError, match="no score"):
        T.rank(scores, labels, keys=["1:0xaa", "1:0xbb"])


def test_ranking_refuses_an_unlabelled_hash():
    with pytest.raises(EvaluationError, match="no label"):
        T.rank({"1:0xaa": 1.0}, {})


def test_k_beyond_the_population_is_refused():
    ordered, _, _ = build({"aa": (1.0, 1)})
    with pytest.raises(EvaluationError):
        T.top_k(ordered, 2, method="m")
    with pytest.raises(EvaluationError):
        T.top_k(ordered, 0, method="m")


def test_overlap_adds_the_tied_block_view_when_the_second_method_ties():
    first, _, _ = build({"aa": (9.0, 1), "bb": (8.0, 1), "cc": (7.0, 0)})
    scores = {f"1:0x{h}": 5.0 for h in ("cc", "dd", "ee", "ff")}
    scores["1:0xaa"] = 9.0
    labels = {"1:0xaa": 1, "1:0xcc": 0, "1:0xdd": 1, "1:0xee": 0, "1:0xff": 0}
    second_full = T.rank(scores, labels)
    second = second_full[:2]                          # aa, then one of the tie
    out = T.overlap(first[:3], second, b_full=second_full)
    block = out["second_method_tied_block"]
    assert block["size"] == 4 and block["score"] == 5.0
    # The realised overlap can be small while the pool agreement is larger.
    assert block["shared_with_block"] >= out["shared"] - 1
    assert out["shared"] >= 1                         # aa is in both


def test_overlap_omits_the_block_view_when_there_is_no_tie():
    first, _, _ = build({"aa": (9.0, 1), "bb": (8.0, 0)})
    second, _, _ = build({"aa": (9.0, 1), "bb": (8.0, 0)})
    out = T.overlap(first[:1], second[:1], b_full=second)
    assert "second_method_tied_block" not in out
    assert out["shared"] == 1 and out["jaccard"] == 1.0


def test_average_precision_needs_both_classes():
    ordered, _, _ = build({"aa": (9.0, 1), "bb": (8.0, 1)})
    assert T.average_precision(ordered) is None
    ordered, _, _ = build({"aa": (9.0, 1), "bb": (8.0, 0)})
    assert T.average_precision(ordered) == pytest.approx(1.0)


def test_selection_rows_carry_rank_score_and_outcome():
    ordered, _, _ = build({"aa": (9.0, 1), "bb": (8.0, 0)})
    rows = T.selection_rows(T.top_k(ordered, 2, method="m"))
    assert [r["rank"] for r in rows] == [1, 2]
    assert [r["outcome"] for r in rows] == ["attack", "false_alert"]
    assert rows[0]["score"] == 9.0 and rows[0]["k"] == 2
