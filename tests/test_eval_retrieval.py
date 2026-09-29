"""BM25 retrieval over a question's candidate docs."""

from laya_compactor.eval.retrieval import retrieve


DOCS = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "Gustave Eiffel's company designed the tower for the 1889 World's Fair.",
    "The Mediterranean diet emphasizes olive oil and fresh vegetables.",
]


def test_retrieve_ranks_matching_docs_above_unrelated_ones():
    top = retrieve("Who designed the Eiffel Tower?", DOCS, k=2)
    assert len(top) == 2
    assert any("Eiffel" in doc for doc in top)
    assert "Photosynthesis" not in " ".join(top)


def test_retrieve_caps_at_k_and_orders_best_first():
    top = retrieve("Gustave Eiffel designed the tower for the 1889 World's Fair",
                   DOCS, k=10)
    assert len(top) == 4  # never more than the candidate set
    # the doc sharing the most query terms leads (on a corpus this tiny some
    # terms hit Okapi's zero-idf edge case, so only the decisive doc is asserted)
    assert top[0].startswith("Gustave")


def test_retrieve_with_empty_candidates_returns_empty():
    assert retrieve("anything", [], k=5) == []
