"""The packaged mini-dataset: 100 published, hand-labeled query/doc rows."""

from collections import Counter

from laya_compactor.sensitivity import load_dataset

EXPECTED_LABELS = {0, 1, 2, 3}


def test_mini_dataset_has_100_rows_with_valid_schema():
    rows = load_dataset()

    assert len(rows) == 100
    for row in rows:
        assert set(row) == {"id", "query", "doc", "label", "rationale"}
        assert isinstance(row["query"], str) and row["query"]
        assert isinstance(row["doc"], str) and row["doc"]
        assert isinstance(row["rationale"], str) and row["rationale"]
        assert row["label"] in EXPECTED_LABELS


def test_mini_dataset_ids_are_unique_and_sequential():
    rows = load_dataset()
    ids = [row["id"] for row in rows]

    assert len(set(ids)) == 100
    assert ids == sorted(ids)  # rel-001..rel-100 in order


def test_mini_dataset_covers_all_four_labels():
    counts = Counter(row["label"] for row in load_dataset())

    assert set(counts) == EXPECTED_LABELS
    # roughly balanced: no label under a fifth of the dataset
    assert all(count >= 20 for count in counts.values()), counts
