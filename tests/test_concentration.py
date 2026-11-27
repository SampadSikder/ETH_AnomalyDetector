"""A per-class miss has to be attributable: few incidents, or a weak signal."""
from __future__ import annotations

import pandas as pd
import pytest

from txbench import concentration as C
from txbench.errors import EvaluationError


def manifest_frame(rows):
    return pd.DataFrame(rows)


def kc_rows(incident_counts: dict[str, int]):
    rows = []
    block = 100
    for incident, count in incident_counts.items():
        for i in range(count):
            block += 1
            rows.append({
                "tx_key": f"1:0x{incident}{i:04d}", "attack_class": "key_compromise",
                "group_incident": incident, "group_protocol": f"P{incident}",
                "block_number": block, "block_timestamp": "2025-08-14T00:00:00Z",
                "split_temporal": "test", "label": 1,
            })
    return rows


def test_one_dominant_incident_is_reported_as_concentration():
    conc = C.concentration_of(manifest_frame(kc_rows({"A": 153, "B": 5})),
                              "key_compromise", partition="test")
    assert conc.transactions == 158
    assert conc.incidents == 2
    assert conc.largest_incident == "A"
    assert conc.largest_incident_transactions == 153
    assert conc.largest_incident_share == pytest.approx(153 / 158)
    # One event dominating pushes the index toward 1; an even split sits at 1/n.
    assert conc.hhi > 0.9
    assert conc.even_hhi == pytest.approx(0.5)


def test_evenly_spread_incidents_have_the_minimum_index():
    conc = C.concentration_of(manifest_frame(kc_rows({"A": 10, "B": 10, "C": 10, "D": 10})),
                              "key_compromise", partition="test")
    assert conc.incidents == 4
    assert conc.hhi == pytest.approx(conc.even_hhi)
    assert conc.largest_incident_share == pytest.approx(0.25)


def test_a_class_with_no_transactions_is_refused_not_reported_as_zero():
    with pytest.raises(EvaluationError):
        C.concentration_of(manifest_frame(kc_rows({"A": 3})), "exploit", partition="test")


def test_shape_comes_from_the_matrix_the_model_saw():
    matrix = pd.DataFrame({
        "tx_key": ["1:0xa", "1:0xb", "1:0xc"],
        C.SEL_ERC20_TRANSFER: [1, 0, 1],
        "sel:0x095ea7b3": [0, 0, 1],
        "sel:0x23b872dd": [0, 0, 1],
        "sel:0x2e1a7d4d": [0, 0, 1],
        "tpc0:0xddf2": [1, 0, 1],
    })
    shapes = C.shapes_from_matrix(matrix)
    by_key = dict(zip(shapes["tx_key"], shapes["shape"]))
    assert by_key["1:0xa"] == C.SHAPE_PLAIN_ERC20       # transfer + its event
    assert by_key["1:0xb"] == C.SHAPE_BARE_VALUE        # nothing fires at all
    assert by_key["1:0xc"] == C.SHAPE_OTHER             # too many tokens to be a control


def test_a_matrix_without_token_columns_is_refused():
    with pytest.raises(EvaluationError):
        C.shapes_from_matrix(pd.DataFrame({"tx_key": ["1:0xa"], "label": [1]}))


def _fixture():
    scores = {"1:0xk1": 0.10, "1:0xk2": 0.11,          # key compromises
              "1:0xn1": 0.10, "1:0xn2": 0.12,          # same-shape negatives
              "1:0xn3": 0.90}                          # a complex negative
    labels = {"1:0xk1": 1, "1:0xk2": 1, "1:0xn1": 0, "1:0xn2": 0, "1:0xn3": 0}
    classes = {"1:0xk1": "key_compromise", "1:0xk2": "key_compromise"}
    shapes = {"1:0xk1": C.SHAPE_PLAIN_ERC20, "1:0xk2": C.SHAPE_PLAIN_ERC20,
              "1:0xn1": C.SHAPE_PLAIN_ERC20, "1:0xn2": C.SHAPE_PLAIN_ERC20,
              "1:0xn3": C.SHAPE_OTHER}
    return scores, labels, classes, shapes


def test_contrast_matches_on_shape_and_leaves_the_threshold_alone():
    scores, labels, classes, shapes = _fixture()
    out = C.contrast(scores, labels, classes, shapes, attack_class="key_compromise",
                     threshold=0.5, keys=list(scores))
    assert out["threshold"] == 0.5                      # handed in, not re-chosen
    assert out["class_scores"]["flagged"] == 0
    assert out["dominant_shape"] == C.SHAPE_PLAIN_ERC20
    matched = out["shape_matched"]
    assert matched["negatives"]["n"] == 2               # the complex negative is excluded
    # Inside one shape the two sides interleave, so the ranking is uninformative.
    assert matched["auc"] == pytest.approx(0.5, abs=0.3)
    # Against the complex negative the class looks *less* suspicious, which is
    # the confound the matched contrast exists to remove.
    assert out["by_negative_group"][C.SHAPE_OTHER]["auc_class_over_group"] == 0.0


def test_contrast_says_so_when_no_negative_shares_the_shape():
    scores, labels, classes, shapes = _fixture()
    shapes["1:0xn1"] = shapes["1:0xn2"] = C.SHAPE_OTHER
    out = C.contrast(scores, labels, classes, shapes, attack_class="key_compromise",
                     threshold=0.5, keys=list(scores))
    assert out["shape_matched"] is None
    assert "shape_matched_absent" in out


def test_auc_counts_ties_as_half():
    assert C._auc([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)
    assert C._auc([2.0], [1.0]) == 1.0
    assert C._auc([], [1.0]) is None


def test_incident_recall_counts_events_not_transactions():
    scores = {f"1:0xa{i}": 0.1 for i in range(10)}
    scores["1:0xb0"] = 0.9
    classes = {k: "key_compromise" for k in scores}
    incidents = {k: ("A" if k.startswith("1:0xa") else "B") for k in scores}
    out = C.incident_recall(scores, classes, incidents,
                            attack_class="key_compromise", threshold=0.5)
    assert out["transactions_scored"] == 11
    assert out["transactions_detected"] == 1
    assert out["incidents"] == 2
    assert out["incidents_detected"] == 1
    # 1 of 11 transactions, but 1 of 2 events: the second number is the honest one.
    assert out["incident_recall"] == pytest.approx(0.5)


def test_contrast_reports_the_sets_own_negative_categories_too():
    scores, labels, classes, shapes = _fixture()
    categories = {"1:0xn1": "ordinary", "1:0xn2": "ordinary", "1:0xn3": "behavioral_hard"}
    out = C.contrast(scores, labels, classes, shapes, attack_class="key_compromise",
                     threshold=0.5, keys=list(scores), categories=categories)
    # How a negative was sampled and what shape it has are different questions,
    # so both groupings are present and neither replaces the other.
    assert set(out["by_negative_category"]) == {"ordinary", "behavioral_hard"}
    assert out["by_negative_category"]["ordinary"]["n"] == 2
    assert out["by_negative_category"]["behavioral_hard"]["flagged"] == 1
    assert out["by_negative_category"]["behavioral_hard"]["auc_class_over_group"] == 0.0
    assert out["shape_matched"]["negatives"]["n"] == 2


def test_contrast_omits_the_category_grouping_when_none_is_supplied():
    scores, labels, classes, shapes = _fixture()
    out = C.contrast(scores, labels, classes, shapes, attack_class="key_compromise",
                     threshold=0.5, keys=list(scores))
    assert "by_negative_category" not in out
