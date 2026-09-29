"""LangChain integration: query-aware document compressor.

Plugs into ContextualCompressionRetriever so any LangChain retriever gets
laya-compactor's budgeted relevance selection as its compression step.
"""

import asyncio
from typing import Any, Callable, Optional, Sequence

from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document

from laya_compactor.core import CompactResult, compact, default_token_counter


class LayaCompactor(BaseDocumentCompressor):
    """Keep the most query-relevant Documents that fit a token budget."""

    budget: int
    agent: Any = None  # inject a fake in tests; None lazily loads the real checkpoint
    min_score: float = 1.0
    token_counter: Optional[Callable[[str], int]] = None
    last_result: Any = None  # CompactResult of the last call (stats + cut reasons)

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        counter = self.token_counter or default_token_counter
        result = compact(
            query,
            [d.page_content for d in documents],
            self.budget,
            agent=self.agent,
            min_score=self.min_score,
            token_counter=counter,
        )
        self.last_result = result
        by_index = dict(enumerate(documents))
        return [
            Document(
                page_content=s.doc,
                metadata={
                    **by_index[s.index].metadata,
                    "laya_score": s.score,
                    "laya_tokens": s.tokens,
                },
            )
            for s in result.kept
        ]

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks=None,
    ) -> Sequence[Document]:
        # the forward pass is blocking CPU work: keep it off the event loop
        return await asyncio.to_thread(
            self.compress_documents, documents, query, callbacks)
