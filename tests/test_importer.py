"""Cohort import: labels exactly as supplied, and nothing inferred."""
from __future__ import annotations

import pytest

from conftest import H, REPO


def test_fixture_is_marked_as_fixture(imported):
    assert imported.meta["is_fixture"] is True
    assert imported.manifest["is_fixture"].all()


def test_chain_filter_counts_before_and_after(imported):
    steps = {s["filter"]: s for s in imported.meta["filters"]}
    chain = steps["chain_id_eq_1"]
    assert chain["rows_in"] > chain["rows_out"]
    assert chain["detail"]["rows_by_other_chain_id"] == {"56": 1}
    # Every filter reports both sides.
    for step in imported.meta["filters"]:
        assert step["rows_in"] >= step["rows_out"] >= 0


def test_identical_duplicate_is_collapsed_with_provenance(manifest):
    row = manifest[manifest["tx_hash"] == H(9)]
    assert len(row) == 1, "an identical duplicate must collapse to one row"
    assert int(row.iloc[0]["duplicate_count"]) == 2
    assert "," in row.iloc[0]["source_rows"], "both source rows are recorded"


def test_conflicting_duplicate_is_quarantined_and_never_benign(imported):
    conflicting = imported.quarantine[imported.quarantine["reason"] == "conflicting_duplicate"]
    assert len(conflicting) == 2
    assert H(10) not in set(imported.manifest["tx_hash"]), \
        "a conflicting duplicate must not enter the manifest at all"
    assert set(conflicting["raw_label"]) == {"0", "1"}


def test_unrecognised_label_is_quarantined_not_coerced(imported):
    bad = imported.quarantine[imported.quarantine["reason"] == "label_unrecognized"]
    assert len(bad) == 1
    assert H(11) not in set(imported.manifest["tx_hash"])
    # and it certainly did not become a negative
    assert not (imported.manifest["tx_hash"] == H(11)).any()


def test_invalid_hash_is_quarantined(imported):
    bad = imported.quarantine[imported.quarantine["reason"] == "invalid_tx_hash"]
    assert len(bad) == 1


def test_labels_are_imported_verbatim(manifest):
    attack = manifest[manifest["tx_hash"] == H(1)].iloc[0]
    assert int(attack["label"]) == 1
    assert attack["attack_class"] == "exploit"
    import pandas as pd
    assert pd.isna(attack["benign_category"]), "an attack row carries no benign category"
    key = manifest[manifest["tx_hash"] == H(7)].iloc[0]
    assert bool(key["is_key_compromise"]) is True


def test_upstream_splits_are_preserved(manifest):
    for column in ("split_temporal", "split_incident", "split_protocol", "split_address",
                   "split_cross_chain"):
        assert column in manifest.columns


def test_cohort_digest_is_deterministic(release):
    from txbench.importer import import_labels
    first = import_labels(release, chain_id=1)
    second = import_labels(release, chain_id=1)
    assert first.cohort_sha256 == second.cohort_sha256
    assert list(first.manifest["tx_key"]) == sorted(first.manifest["tx_key"])


def test_digest_changes_when_a_label_changes(imported):
    from txbench.hashutil import cohort_digest
    from txbench.importer import DIGEST_FIELDS
    rows = imported.manifest.to_dict(orient="records")
    before = cohort_digest(rows, DIGEST_FIELDS)
    rows[0] = {**rows[0], "label": 1 - int(rows[0]["label"])}
    assert cohort_digest(rows, DIGEST_FIELDS) != before


def test_manifest_round_trips(imported, tmp_path):
    from txbench.importer import load_manifest, write_manifest
    written = write_manifest(imported, manifest_path=tmp_path / "m.parquet",
                             meta_path=tmp_path / "m.json",
                             mapping_path=tmp_path / "map.json",
                             quarantine_path=tmp_path / "q.parquet")
    reloaded = load_manifest(written["manifest"])
    assert list(reloaded["tx_key"]) == list(imported.manifest["tx_key"])


def test_the_shipped_config_pins_the_release_doi():
    """The DOI must survive a plain `bench import-txbench` with no --doi flag.

    It was once supplied only by the CLI flag, so a bare re-import overwrote the
    recorded DOI with null and lost provenance section 1 requires.
    """
    from txbench.config import load_config

    cfg = load_config(REPO / "config" / "benchmark.toml")
    assert cfg.release_doi, "cohort.release_doi is not pinned in config/benchmark.toml"
    assert cfg.release_doi.startswith("10."), cfg.release_doi
    assert cfg.effective_settings()["release_doi"] == cfg.release_doi
