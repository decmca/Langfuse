# (See earlier content) — Full file inserted
"""
Retrieval component for RAG pipeline with vector database integration.

Supports ChromaDB for efficient similarity search over document collections.

Author: Declan McAlinden
Date: 2025-11-10
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class Retriever:
    """
    Document retrieval component using vector similarity search.
    
    This class handles:
    - Building document index from text chunks
    - Efficient similarity search using ChromaDB
    - Top-k retrieval with metadata filtering
    - Relevance score thresholding
    
    Attributes:
        embedder: Embedding model for vectorisation
        collection: ChromaDB collection for storage
        top_k (int): Number of documents to retrieve per query
        min_similarity (float): Minimum similarity threshold for retrieval
    """
    
    def __init__(
        self,
        embedder,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        top_k: int = 5,
        min_similarity: float = 0.0
    ):
        """
        Initialise retriever with embedding model and vector database.
        
        Args:
            embedder: Embedder instance for generating document vectors
            collection_name: Name of ChromaDB collection
            persist_directory: Directory for persisting vector database
            top_k: Number of top documents to retrieve
            min_similarity: Minimum similarity score threshold (0-1)
        """
        self.embedder = embedder
        self.top_k = top_k
        self.min_similarity = min_similarity
        
        logger.info(f"Initialising ChromaDB (collection: {collection_name})")
        
        # Initialise ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection with {self.collection.count()} documents")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info("Created new collection")
    
    def index_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
        show_progress: bool = True
    ) -> None:
        """
        Index a collection of documents into the vector database.
        
        Documents are embedded and stored with their metadata for efficient retrieval.
        
        Args:
            documents: List of document text chunks to index
            metadatas: Optional list of metadata dicts (one per document)
            batch_size: Number of documents to process in each batch
            show_progress: Whether to display progress information
        
        Raises:
            ValueError: If documents list is empty
        """
        if not documents:
            raise ValueError("Cannot index empty document list")
        
        num_docs = len(documents)
        logger.info(f"Indexing {num_docs} documents...")
        
        # Generate embeddings
        embeddings = self.embedder.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress
        )
        
        # Prepare metadatas (use empty dict if not provided)
        if metadatas is None:
            metadatas = [{}] * num_docs
        
        # Add to ChromaDB in batches
        for i in range(0, num_docs, batch_size):
            end_idx = min(i + batch_size, num_docs)
            
            self.collection.add(
                ids=[f"doc_{j}" for j in range(i, end_idx)],
                embeddings=embeddings[i:end_idx].tolist(),
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            if show_progress:
                logger.info(f"Indexed {end_idx}/{num_docs} documents")
        
        logger.info("Document indexing complete")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_meta Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant documents for a given query.
        
        Args:
            query: Query text string
            top_k: Number of documents to retrieve (overrides default)
            filter_meta Optional metadata filters (e.g., {"source": "pubmed"})
        
        Returns:
            List of retrieved documents with metadata and relevance scores.
            Each item is a dictionary containing:
            - "text": Document text
            - "score": Relevance score (0-1)
            - "metadata": Document metadata
            - "id": Document identifier
        """
        k = top_k if top_k is not None else self.top_k
        
        # Encode query
        query_embedding = self.embedder.encode_query(query)
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=filter_metadata
        )
        
        # Format results
        retrieved_docs = []
        
        if results["documents"] and results["documents"][0]:
            for i, doc_text in enumerate(results["documents"][0]):
                # Convert distance to similarity score (ChromaDB returns cosine distance)
                # Cosine similarity = 1 - cosine distance
                similarity = 1.0 - results["distances"][0][i]
                
                # Filter by minimum similarity
                if similarity >= self.min_similarity:
                    retrieved_docs.append({
                        "text": doc_text,
                        "score": float(similarity),
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "id": results["ids"][0][i]
                    })
        
        logger.debug(f"Retrieved {len(retrieved_docs)} documents for query")
        
        return retrieved_docs
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Collection cleared")
