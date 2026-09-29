"""LlamaIndex integration: node postprocessor.

Drop it into any query engine pipeline: `node_postprocessors=[
LayaCompactorPostprocessor(budget=1500)]`.
"""

from typing import Any, Callable, List, Optional

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from laya_compactor.core import compact, default_token_counter


class LayaCompactorPostprocessor(BaseNodePostprocessor):
    """Keep the most query-relevant nodes that fit a token budget."""

    budget: int
    agent: Any = None  # inject a fake in tests; None lazily loads the real checkpoint
    min_score: float = 1.0
    token_counter: Optional[Callable[[str], int]] = None

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            # nothing to score against: keep everything that fits, in order
            counter = self.token_counter or default_token_counter
            kept, tokens = [], 0
            for node in nodes:
                n = counter(node.node.get_content())
                if tokens + n > self.budget:
                    continue
                kept.append(node)
                tokens += n
            return kept
        counter = self.token_counter or default_token_counter
        result = compact(
            query_bundle.query_str,
            [n.node.get_content() for n in nodes],
            self.budget,
            agent=self.agent,
            min_score=self.min_score,
            token_counter=counter,
        )
        by_index = dict(enumerate(nodes))
        return [by_index[s.index] for s in result.kept]
