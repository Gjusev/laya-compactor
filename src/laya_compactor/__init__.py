"""laya-compactor: context compaction for RAG and agents: score every retrieved chunk in one shared forward pass and cut what does not matter before it reaches the LLM."""

from laya_compactor.core import (
    CompactResult,
    CompactStats,
    CutDoc,
    ScoredDoc,
    compact,
    default_token_counter,
)
from laya_compactor.rubrics import RELEVANCE_RUBRIC, RUBRIC_VARIANTS, get_rubric
from laya_compactor.sensitivity import load_dataset, run_sensitivity

__version__ = "0.1.0"

__all__ = [
    "CompactResult",
    "CompactStats",
    "CutDoc",
    "ScoredDoc",
    "RELEVANCE_RUBRIC",
    "RUBRIC_VARIANTS",
    "compact",
    "default_token_counter",
    "get_rubric",
    "load_dataset",
    "run_sensitivity",
]
