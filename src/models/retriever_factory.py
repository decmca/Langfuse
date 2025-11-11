"""
Factory pattern for creating retriever instances.

Provides flexible switching between different vector database backends
based on configuration.

Author: Declan McAlinden
Date: 2025-11-11
"""

import logging
import os
from typing import Optional
from .retriever import Retriever
from .retriever_lance import LanceRetriever

logger = logging.getLogger(__name__)


def create_retriever(
    embedder,
    vector_store_type: Optional[str] = None,
    collection_name: str = "rag_documents",
    persist_directory: Optional[str] = None,
    top_k: int = 5,
    min_similarity: float = 0.0,
    distance_type: str = "cosine"
):
    """
    Factory function to create retriever instances.
    
    Args:
        embedder: Embedder instance for generating document vectors
        vector_store_type: Type of vector store ("chromadb" or "lancedb").
                          If None, reads from VECTOR_STORE_TYPE env variable
        collection_name: Name of the collection/table
        persist_directory: Directory for persisting vector database.
                          If None, uses default based on store type
        top_k: Number of top documents to retrieve
        min_similarity: Minimum similarity score threshold
        distance_type: Distance metric (for LanceDB: "cosine", "l2", "dot")
    
    Returns:
        Retriever instance (either Retriever or LanceRetriever)
    
    Raises:
        ValueError: If invalid vector_store_type specified
    """
    # Determine vector store type
    if vector_store_type is None:
        vector_store_type = os.getenv("VECTOR_STORE_TYPE", "chromadb").lower()
    
    vector_store_type = vector_store_type.lower()
    
    # Set default persist directory based on store type
    if persist_directory is None:
        if vector_store_type == "lancedb":
            persist_directory = os.getenv("LANCE_DB_DIR", "./lance_db")
        else:
            persist_directory = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    
    logger.info(f"Creating retriever with backend: {vector_store_type}")
    
    # Create appropriate retriever
    if vector_store_type == "lancedb":
        return LanceRetriever(
            embedder=embedder,
            table_name=collection_name,
            persist_directory=persist_directory,
            top_k=top_k,
            min_similarity=min_similarity,
            distance_type=distance_type
        )
    elif vector_store_type == "chromadb":
        return Retriever(
            embedder=embedder,
            collection_name=collection_name,
            persist_directory=persist_directory,
            top_k=top_k,
            min_similarity=min_similarity
        )
    else:
        raise ValueError(
            f"Invalid vector_store_type: {vector_store_type}. "
            "Must be 'chromadb' or 'lancedb'"
        )
