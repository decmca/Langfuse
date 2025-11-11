"""
Factory pattern for creating retriever instances.

Provides LanceDB-based retrieval with optional future extensibility.

Author: Declan McAlinden
Date: 2025-11-11
"""

import logging
import os
from typing import Optional
from .retriever_lance import LanceRetriever

logger = logging.getLogger(__name__)


def create_retriever(
    embedder,
    collection_name: str = "rag_documents",
    persist_directory: Optional[str] = None,
    top_k: int = 5,
    min_similarity: float = 0.0,
    distance_type: str = "cosine"
):
    """
    Factory function to create LanceDB retriever instances.
    
    Args:
        embedder: Embedder instance for generating document vectors
        collection_name: Name of the table
        persist_directory: Directory for persisting vector database.
                          If None, uses LANCE_DB_DIR from environment
        top_k: Number of top documents to retrieve
        min_similarity: Minimum similarity score threshold
        distance_type: Distance metric ("cosine", "l2", "dot")
    
    Returns:
        LanceRetriever instance
    """
    # Set default persist directory
    if persist_directory is None:
        persist_directory = os.getenv("LANCE_DB_DIR", "./lancedb")
    
    logger.info(f"Creating LanceDB retriever at {persist_directory}")
    
    return LanceRetriever(
        embedder=embedder,
        table_name=collection_name,
        persist_directory=persist_directory,
        top_k=top_k,
        min_similarity=min_similarity,
        distance_type=distance_type
    )
