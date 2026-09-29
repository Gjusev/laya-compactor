"""Opt-in tests that load the real laya checkpoint.

Skipped by default (addopts deselects them); run with: pytest -m slow
The first run downloads the checkpoint from the Hugging Face Hub.
"""

import pytest

from laya_compactor.core import compact
from laya_compactor.sensitivity import load_dataset, run_sensitivity

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def agent():
    import laya

    return laya.load()


def test_real_agent_compacts_a_small_batch(agent):
    docs = [
        "The Eiffel Tower is a wrought-iron lattice tower in Paris, completed in 1889.",
        "Photosynthesis lets plants convert sunlight into chemical energy.",
        "The tower was designed by Gustave Eiffel's engineering firm for the 1889 World's Fair.",
    ]
    result = compact("Who designed the Eiffel Tower?", docs, budget=50, agent=agent)

    assert 0 < len(result.kept) <= 3
    assert result.stats.docs_kept + result.stats.docs_cut == 3
    kept_docs = " ".join(d.doc for d in result.kept)
    assert "Eiffel" in kept_docs  # the essential doc survives


def test_real_agent_rubric_sensitivity_on_mini_dataset(agent):
    rows = load_dataset()
    report = run_sensitivity(rows, agent=agent)

    assert report.n == len(rows) == 100
    # sanity only: scores stay inside the 0..3 scale for every variant
    for metrics in report.per_variant.values():
        assert 0.0 <= metrics["mean_score"] <= 3.0
