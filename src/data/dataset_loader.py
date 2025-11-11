"""
Unified dataset loading and preprocessing for RAG evaluation.

Supports SQuAD, MS MARCO, and PubMedQA datasets with consistent interfaces
for easy switching between benchmarks.

Author: Declan McAlinden
Date: 2025-11-10
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datasets import load_dataset, Dataset
import yaml

logger = logging.getLogger(__name__)


@dataclass
class RAGExample:
    """
    Standardised data structure for RAG evaluation examples.
    
    Attributes:
        id: Unique identifier for the example
        question: User query or question text
        contexts: List of relevant context passages
        answer: Ground truth answer text
    metadata Additional metadata (e.g., source, difficulty level)
    """
    id: str
    question: str
    contexts: List[str]
    answer: str
    metadata: Dict[str, Any]


class DatasetLoader:
    """
    Unified interface for loading and preprocessing QA datasets.
    
    This class handles:
    - Loading datasets from HuggingFace Hub
    - Preprocessing and formatting into standardised RAGExample format
    - Chunking long contexts into retrieval-friendly segments
    - Train/eval splitting with configurable sample limits
    
    Supported datasets:
    - SQuAD (v1.1, v2.0)
    - MS MARCO
    - PubMedQA
    """
    
    def __init__(self, config_path: str):
        """
        Initialise dataset loader with configuration file.
        
        Args:
            config_path: Path to dataset configuration YAML file
        
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.dataset_configs = self.config.get('datasets', {})
        self.preprocessing_config = self.config.get('preprocessing', {})
        
        logger.info(f"Loaded configuration for {len(self.dataset_configs)} datasets")
    
    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: Optional[int] = None
    ) -> List[RAGExample]:
        """
        Load and preprocess a dataset into standardised RAGExample format.
        
        Args:
            dataset_name: Name of dataset ("squad", "msmarco", "pubmedqa")
            split: Dataset split to load ("train", "validation", "test")
            max_samples: Maximum number of samples to load (None for all)
        
        Returns:
            List of RAGExample objects ready for evaluation
        
        Raises:
            ValueError: If dataset_name is not supported
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(
                f"Dataset '{dataset_name}' not found in configuration. "
                f"Supported datasets: {list(self.dataset_configs.keys())}"
            )
        
        dataset_config = self.dataset_configs[dataset_name]
        
        # Load from HuggingFace Hub
        logger.info(f"Loading {dataset_name} dataset (split: {split})...")
        
        hf_dataset = self._load_from_hub(dataset_name, dataset_config, split)
        
        # Apply sample limit if specified
        if max_samples:
            hf_dataset = hf_dataset.select(range(min(len(hf_dataset), max_samples)))
        
        logger.info(f"Loaded {len(hf_dataset)} examples")
        
        # Preprocess into standardised format
        examples = self._preprocess_dataset(hf_dataset, dataset_name, dataset_config)
        
        return examples
    
    def _load_from_hub(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        split: str
    ) -> Dataset:
        """
        Load dataset from HuggingFace Hub with error handling.
        
        Args:
            dataset_name: Name of the dataset
            config: Dataset configuration dictionary
            split: Split to load
        
        Returns:
            HuggingFace Dataset object
        
        Raises:
            RuntimeError: If dataset loading fails
        """
        try:
            hf_name = config.get("name", dataset_name)
            hf_config = config.get("config", None)
            trust_remote_code = config.get("trust_remote_code", False)
            
            # Map split names
            split_mapping = {
                "train": config.get("train_split", "train"),
                "validation": config.get("eval_split", "validation"),
                "test": config.get("test_split", "test")
            }
            hf_split = split_mapping.get(split, split)
            
            if hf_config:
                dataset = load_dataset(
                    hf_name,
                    hf_config,
                    split=hf_split,
                    trust_remote_code=trust_remote_code
                )
            else:
                dataset = load_dataset(hf_name, split=hf_split)
            
            return dataset
        
        except Exception as e:
            raise RuntimeError(f"Failed to load {dataset_name}: {str(e)}")
    
    def _preprocess_dataset(
        self,
        dataset: Dataset,
        dataset_name: str,
        config: Dict[str, Any]
    ) -> List[RAGExample]:
        """
        Preprocess dataset into standardised RAGExample format.
        
        Args:
            dataset: HuggingFace Dataset object
            dataset_name: Name of the dataset for processor selection
            config: Dataset configuration dictionary
        
        Returns:
            List of RAGExample objects
        """
        if dataset_name == "squad":
            return self._preprocess_squad(dataset, config)
        elif dataset_name == "msmarco":
            return self._preprocess_msmarco(dataset, config)
        elif dataset_name == "pubmedqa":
            return self._preprocess_pubmedqa(dataset, config)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _preprocess_squad(
        self,
        dataset: Dataset,
        config: Dict[str, Any]
    ) -> List[RAGExample]:
        """
        Preprocess SQuAD dataset.
        
        SQuAD format:
        - context: Long paragraph containing answer
        - question: Question about the context
        - answers: List of answer spans (text and start position)
        
        Args:
            dataset: SQuAD dataset
            config: SQuAD configuration
        
        Returns:
            List of RAGExample objects with chunked contexts
        """
        examples = []
        
        for idx, item in enumerate(dataset):
            context = item[config.get("context_key", "context")]
            question = item[config.get("question_key", "question")]
            answers = item[config.get("answer_key", "answers")]
            
            # Extract answer text (handle SQuAD 2.0 format)
            if answers and "text" in answers and len(answers["text"]) > 0:
                answer_text = answers["text"][0]
            else:
                answer_text = ""  # Unanswerable question in SQuAD 2.0
            
            # Chunk context into retrievable segments
            chunks = self._chunk_text(context)
            
            examples.append(RAGExample(
                id=item.get("id", f"squad_{idx}"),
                question=question,
                contexts=chunks,
                answer=answer_text,
                metadata={
                    "dataset": "squad",
                    "full_context": context,
                    "is_answerable": bool(answer_text)
                }
            ))
        
        return examples
    
    def _preprocess_msmarco(
        self,
        dataset: Dataset,
        config: Dict[str, Any]
    ) -> List[RAGExample]:
        """
        Preprocess MS MARCO dataset.
        
        MS MARCO format:
        - query: User search query
        - passages: List of candidate passages with relevance scores
        - answers: List of human-generated answers
        
        Args:
            dataset: MS MARCO dataset
            config: MS MARCO configuration
        
        Returns:
            List of RAGExample objects
        """
        examples = []
        
        for idx, item in enumerate(dataset):
            query = item[config.get("query_key", "query")]
            passages = item.get(config.get("passages_key", "passages"), [])
            
            # Extract passage texts
            contexts = [p.get("passage_text", "") for p in passages if "passage_text" in p]
            
            # Get answer (use first answer if multiple exist)
            answers = item.get("answers", [])
            answer_text = answers[0] if answers else ""
            
            examples.append(RAGExample(
                id=item.get("query_id", f"msmarco_{idx}"),
                question=query,
                contexts=contexts,
                answer=answer_text,
                metadata={
                    "dataset": "msmarco",
                    "num_passages": len(contexts)
                }
            ))
        
        return examples
    
    def _preprocess_pubmedqa(
        self,
        dataset: Dataset,
        config: Dict[str, Any]
    ) -> List[RAGExample]:
        """
        Preprocess PubMedQA dataset.
        
        PubMedQA format:
        - question: Research question
        - context: Concatenated abstracts from PubMed
        - long_answer: Detailed answer explanation
        
        Args:
            dataset: PubMedQA dataset
            config: PubMedQA configuration
        
        Returns:
            List of RAGExample objects
        """
        examples = []
        
        for idx, item in enumerate(dataset):
            question = item[config.get("question_key", "question")]
            context = item.get(config.get("context_key", "context"), {})
            
            # PubMedQA context is often a dict with sections
            if isinstance(context, dict):
                context_text = " ".join([
                    f"{k}: {v}" for k, v in context.items() if v
                ])
            else:
                context_text = str(context)
            
            answer = item.get(config.get("answer_key", "long_answer"), "")
            
            # Chunk context
            chunks = self._chunk_text(context_text)
            
            examples.append(RAGExample(
                id=item.get("pubid", f"pubmed_{idx}"),
                question=question,
                contexts=chunks,
                answer=answer,
                metadata={
                    "dataset": "pubmedqa",
                    "full_context": context_text
                }
            ))
        
        return examples
    
    def _chunk_text(self, text: str) -> List[str]:
        """
        Chunk long text into overlapping segments for retrieval.
        
        Uses sliding window with configurable chunk size and overlap.
        
        Args:
            text: Long text string to chunk
        
        Returns:
            List of text chunks
        """
        chunk_size = self.preprocessing_config.get("chunk_size", 512)
        chunk_overlap = self.preprocessing_config.get("chunk_overlap", 50)
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk:  # Skip empty chunks
                chunks.append(chunk)
        
        return chunks if chunks else [text]  # Return original if chunking fails
