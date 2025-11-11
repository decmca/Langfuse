"""Model components for RAG pipeline."""

from .embedder import Embedder
from .retriever_lance import LanceRetriever
from .generator import Generator
from .retriever_factory import create_retriever

# Use LanceRetriever as default Retriever
Retriever = LanceRetriever

__all__ = ["Embedder", "Retriever", "LanceRetriever", "Generator", "create_retriever"]
