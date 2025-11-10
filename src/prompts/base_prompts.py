"""
Baseline prompt templates for RAG generation.

Author: Declan McAlinden
Date: 2025-11-10
"""

# Baseline: Simple instruction
BASELINE_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided contexts."""

# Citation-aware: Explicit citation instructions
CITATION_AWARE_PROMPT = """You are a helpful AI assistant that answers questions based on provided contexts.

IMPORTANT INSTRUCTIONS:
1. Only use information from the provided contexts
2. Always cite your sources using [1], [2], etc. to reference specific contexts
3. If the answer is not in the contexts, say "I cannot answer based on the provided information"
4. Be precise and concise in your answers
"""
