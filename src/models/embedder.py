# (See earlier content) — Full file inserted
"""
Embedding model wrapper for consistent vector representations.

Supports sentence-transformers and HuggingFace models with unified interface
for easy swapping between embedding models.

Author: Declan McAlinden
Date: 2025-11-10
"""

import logging
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class Embedder:
    """
    Unified interface for text embedding models.
    
    This class wraps different embedding model implementations (sentence-transformers,
    HuggingFace) into a consistent API for the RAG pipeline.
    
    Attributes:
        model_name (str): Name/path of the embedding model
        model: Loaded embedding model instance
        dimension (int): Embedding vector dimension
        max_seq_length (int): Maximum sequence length for input
        device (str): Device for computation ("cuda" or "cpu")
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        max_seq_length: int = 512
    ):
        """
        Initialise embedding model.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Compute device ("cuda", "cpu", or None for auto-detect)
            max_seq_length: Maximum input sequence length (longer sequences are truncated)
        
        Raises:
            RuntimeError: If model loading fails
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading embedding model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            self.model.max_seq_length = max_seq_length
            self.dimension = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded successfully (dimension: {self.dimension})")
        
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{model_name}': {str(e)}")
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Encode a list of texts into embedding vectors.
        
        Args:
            texts: List of text strings to encode
            batch_size: Batch size for encoding (larger = faster but more memory)
            show_progress_bar: Whether to display progress bar
            normalize_embeddings: Whether to L2-normalise embeddings for cosine similarity
        
        Returns:
            NumPy array of shape (len(texts), dimension) containing embeddings
        
        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Cannot encode empty text list")
        
        logger.debug(f"Encoding {len(texts)} texts (batch_size={batch_size})")
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_query(
        self,
        query: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode a single query string.
        
        Convenience method for encoding individual queries without batching.
        
        Args:
            query: Query text string
            normalize: Whether to normalise the embedding
        
        Returns:
            1D NumPy array of shape (dimension,) containing the query embedding
        """
        embedding = self.encode(
            [query],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=normalize
        )
        
        return embedding[0]  # Return single vector (squeeze batch dimension)
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and documents.
        
        Args:
            query_embedding: Query embedding vector of shape (dimension,)
            document_embeddings: Document embeddings of shape (num_docs, dimension)
        
        Returns:
            NumPy array of shape (num_docs,) containing similarity scores [0, 1]
        """
        # Ensure embeddings are normalised for cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        doc_norms = document_embeddings / (
            np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Cosine similarity = dot product of normalised vectors
        similarities = np.dot(doc_norms, query_norm)
        
        return similarities
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "max_seq_length": self.max_seq_length,
            "device": self.device
        }
