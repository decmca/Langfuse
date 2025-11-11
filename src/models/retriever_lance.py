"""
LanceDB-based retrieval component for RAG pipeline.

Provides high-performance vector similarity search using LanceDB's
serverless architecture and Lance columnar format.

Author: Declan McAlinden
Date: 2025-11-11
"""

import logging
from typing import List, Dict, Any, Optional
import lancedb
import numpy as np

logger = logging.getLogger(__name__)


class LanceRetriever:
    """
    Document retrieval component using LanceDB vector database.
    
    This class provides:
    - High-performance vector similarity search
    - Efficient columnar storage with Lance format
    - Native support for multi-modal data
    - Serverless architecture for scalability
    
    Attributes:
        embedder: Embedding model for vectorisation
        db: LanceDB database connection
        table: LanceDB table for document storage
        top_k (int): Number of documents to retrieve per query
        min_similarity (float): Minimum similarity threshold for retrieval
    """
    
    def __init__(
        self,
        embedder,
        table_name: str = "rag_documents",
        persist_directory: str = "./lance_db",
        top_k: int = 5,
        min_similarity: float = 0.0,
        distance_type: str = "cosine"
    ):
        """
        Initialise LanceDB retriever with embedding model.
        
        Args:
            embedder: Embedder instance for generating document vectors
            table_name: Name of LanceDB table
            persist_directory: Directory for persisting vector database
            top_k: Number of top documents to retrieve
            min_similarity: Minimum similarity score threshold (0-1)
            distance_type: Distance metric ("cosine", "l2", or "dot")
        """
        self.embedder = embedder
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.distance_type = distance_type
        self.table_name = table_name
        
        logger.info(f"Initialising LanceDB (table: {table_name})")
        
        # Connect to LanceDB
        self.db = lancedb.connect(persist_directory)
        
        # Check if table exists
        try:
            self.table = self.db.open_table(table_name)
            count = self.table.count_rows()
            logger.info(f"Loaded existing table with {count} documents")
        except Exception:
            self.table = None
            logger.info("Table will be created on first indexing")
    
    def index_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
        show_progress: bool = True
    ) -> None:
        """
        Index a collection of documents into LanceDB.
        
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
        
        # Prepare data for LanceDB
        # LanceDB expects list of dicts with consistent schema
        data = []
        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            record = {
                "id": f"doc_{i}",
                "text": doc,
                "vector": emb.tolist(),
            }
            
            # Add metadata fields if provided
            if metadatas and i < len(metadatas):
                for key, value in metadatas[i].items():
                    record[key] = value
            
            data.append(record)
        
        # Create or append to table
        if self.table is None:
            # Create new table
            self.table = self.db.create_table(
                self.table_name,
                data=data,
                mode="overwrite"
            )
            logger.info(f"Created new table with {len(data)} documents")
        else:
            # Append to existing table
            self.table.add(data)
            logger.info(f"Added {len(data)} documents to existing table")
        
        logger.info("Document indexing complete")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant documents for a given query.
        
        Args:
            query: Query text string
            top_k: Number of documents to retrieve (overrides default)
            filter_metadata: Optional SQL-like filter string (e.g., "source = 'pubmed'")
        
        Returns:
            List of retrieved documents with metadata and relevance scores.
            Each item is a dictionary containing:
            - "text": Document text
            - "score": Relevance score (0-1)
            - "metadata": Document metadata
            - "id": Document identifier
        
        Raises:
            RuntimeError: If table hasn't been created yet
        """
        if self.table is None:
            raise RuntimeError("Cannot retrieve from non-existent table. Index documents first.")
        
        k = top_k if top_k is not None else self.top_k
        
        # Encode query
        query_embedding = self.embedder.encode_query(query)
        
        # Build search query
        search = self.table.search(query_embedding.tolist())
        
        # Set distance metric
        if self.distance_type == "cosine":
            search = search.metric("cosine")
        elif self.distance_type == "l2":
            search = search.metric("l2")
        elif self.distance_type == "dot":
            search = search.metric("dot")
        
        # Apply filter if provided
        if filter_metadata:
            search = search.where(filter_metadata)
        
        # Execute search
        results = search.limit(k).to_list()
        
        # Format results
        retrieved_docs = []
        
        for result in results:
            # LanceDB returns distance, convert to similarity
            distance = result.get("_distance", 0.0)
            
            # Convert distance to similarity based on metric type
            if self.distance_type == "cosine":
                # Cosine distance is already similarity-like (1 = identical)
                similarity = 1.0 - distance
            elif self.distance_type == "l2":
                # L2 distance: smaller is better, convert to similarity
                similarity = 1.0 / (1.0 + distance)
            else:  # dot product
                # Dot product: higher is better
                similarity = distance
            
            # Filter by minimum similarity
            if similarity >= self.min_similarity:
                # Extract metadata (all fields except special ones)
                metadata = {
                    k: v for k, v in result.items() 
                    if k not in ["id", "text", "vector", "_distance"]
                }
                
                retrieved_docs.append({
                    "text": result["text"],
                    "score": float(similarity),
                    "metadata": metadata,
                    "id": result["id"]
                })
        
        logger.debug(f"Retrieved {len(retrieved_docs)} documents for query")
        
        return retrieved_docs
    
    def clear_collection(self) -> None:
        """Clear all documents from the table."""
        if self.table is not None:
            self.db.drop_table(self.table_name)
            self.table = None
            logger.info("Table cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current table.
        
        Returns:
            Dictionary containing table statistics
        """
        if self.table is None:
            return {"exists": False, "count": 0}
        
        return {
            "exists": True,
            "count": self.table.count_rows(),
            "schema": str(self.table.schema),
        }
