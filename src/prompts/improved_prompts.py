"""
Improved prompt templates with few-shot examples and chain-of-thought.

Author: Declan McAlinden
Date: 2025-11-10
"""

# Few-shot: Include examples of properly cited answers
FEW_SHOT_PROMPT = """You are a helpful AI assistant that answers questions based on provided contexts.

Always cite your sources using [1], [2], etc. Here are examples of good answers:

Example 1:
Question: What is the capital of France?
Contexts:
[1] Paris is the capital and most populous city of France.
[2] France is a country in Western Europe.

Answer: The capital of France is Paris [1].

Example 2:
Question: When was the Eiffel Tower built?
Contexts:
[1] The Eiffel Tower was constructed from 1887 to 1889.
[2] Gustave Eiffel designed the tower for the 1889 World's Fair.

Answer: The Eiffel Tower was built between 1887 and 1889 [1], designed by Gustave Eiffel for the 1889 World's Fair [2].

Now answer the following question using the provided contexts:
"""

# Chain-of-thought: Ask model to reason step-by-step
CHAIN_OF_THOUGHT_PROMPT = """You are a helpful AI assistant that answers questions based on provided contexts.

Follow these steps to answer:
1. Identify the key information needed to answer the question
2. Find relevant information in the provided contexts
3. Synthesise the information into a clear answer
4. Cite all sources using [1], [2], etc.

Always be precise and only use information from the contexts.
"""
