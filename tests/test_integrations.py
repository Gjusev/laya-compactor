"""Integration wrappers, tested against the real frameworks with dummy
retrievers and the fake laya agent (no checkpoint, no network)."""

import pytest

from laya_compactor.integrations.langchain import LayaCompactor

# -- LangChain ---------------------------------------------------------------

from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.retrievers import BaseRetriever


class DummyRetriever(BaseRetriever):
    """Returns the same three docs for every query, no index."""

    docs: list = None

    def _get_relevant_documents(self, query, *, run_manager=None):
        return [Document(page_content=d, metadata={"i": i}) for i, d in enumerate(self.docs)]

    async def _aget_relevant_documents(self, query, *, run_manager=None):
        return self._get_relevant_documents(query)


DOCS = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The tower was designed by Gustave Eiffel for the 1889 World's Fair.",
]


def make_agent():
    class Agent:
        def predict_batch(self, states, questions):
            return [{"answers": {"relevance": {"type": "score",
                                               "score": 3.0 if "Gustave" in s["document"] else 0.2}}}
                    for s in states]

    return Agent()


def word_counter(text):
    return len(text.split())


def test_compressor_keeps_scoreing_docs_in_relevance_order():
    compressor = LayaCompactor(budget=100, agent=make_agent(),
                               token_counter=word_counter)
    documents = [Document(page_content=d) for d in DOCS]

    kept = compressor.compress_documents(documents, "Who designed the Eiffel Tower?")

    assert [d.page_content for d in kept] == [DOCS[2]]
    assert kept[0].metadata["laya_score"] == pytest.approx(3.0)


def test_compressor_exposes_stats_and_cut_reasons():
    compressor = LayaCompactor(budget=100, agent=make_agent(),
                               token_counter=word_counter)
    compressor.compress_documents([Document(page_content=d) for d in DOCS], "q")
    stats = compressor.last_result.stats
    assert stats.docs_total == 3 and stats.docs_cut == 2
    assert all(r.reason for r in compressor.last_result.cut)


def test_wires_into_contextual_compression_retriever():
    base = DummyRetriever(docs=DOCS)
    retriever = ContextualCompressionRetriever(
        base_retriever=base,
        base_compressor=LayaCompactor(budget=100, agent=make_agent(),
                                      token_counter=word_counter),
    )

    results = retriever.invoke("Who designed the Eiffel Tower?")

    assert [r.page_content for r in results] == [DOCS[2]]


def test_async_compression_matches_sync():
    import asyncio

    compressor = LayaCompactor(budget=100, agent=make_agent(),
                               token_counter=word_counter)
    documents = [Document(page_content=d) for d in DOCS]

    kept = asyncio.run(compressor.acompress_documents(
        documents, "Who designed the Eiffel Tower?"))

    assert [d.page_content for d in kept] == [DOCS[2]]


# -- LlamaIndex --------------------------------------------------------------

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from laya_compactor.integrations.llamaindex import LayaCompactorPostprocessor


def make_nodes():
    return [NodeWithScore(node=TextNode(text=d, metadata={"i": i})) for i, d in enumerate(DOCS)]


def test_postprocessor_keeps_scoreing_nodes_in_relevance_order():
    pp = LayaCompactorPostprocessor(budget=100, agent=make_agent(),
                                    token_counter=word_counter)

    kept = pp.postprocess_nodes(make_nodes(),
                                query_bundle=QueryBundle(query_str="Who designed the Eiffel Tower?"))

    assert [n.node.text for n in kept] == [DOCS[2]]
    assert kept[0].node.metadata["i"] == 2


def test_postprocessor_is_a_real_node_postprocessor():
    assert issubclass(LayaCompactorPostprocessor, BaseNodePostprocessor)


def test_postprocessor_without_query_keeps_everything_within_budget():
    # no query bundle: nothing to score against; keep what fits, in order
    pp = LayaCompactorPostprocessor(budget=100, agent=make_agent(),
                                    token_counter=word_counter)
    kept = pp.postprocess_nodes(make_nodes())

    assert len(kept) == 3
