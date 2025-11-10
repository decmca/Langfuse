"""Prompt templates for RAG generation."""

from .base_prompts import BASELINE_SYSTEM_PROMPT, CITATION_AWARE_PROMPT
from .improved_prompts import FEW_SHOT_PROMPT, CHAIN_OF_THOUGHT_PROMPT

__all__ = [
    "BASELINE_SYSTEM_PROMPT",
    "CITATION_AWARE_PROMPT",
    "FEW_SHOT_PROMPT",
    "CHAIN_OF_THOUGHT_PROMPT"
]
