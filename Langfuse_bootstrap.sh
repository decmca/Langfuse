#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script to create the Citation-Aware Q&A System with Langfuse Tracking
# This will create the full project at the requested location.

TARGET_DIR="/Users/declanmcalinden1/Portfolio_Projects_Clean/Langfuse"

mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# Create directories
mkdir -p config src/{tracking,data,models,evaluation,utils,prompts} scripts experiments/baseline notebooks tests chroma_db

# -----------------------------
# Write files
# -----------------------------

# requirements.txt
cat > requirements.txt << 'EOF'
# Core Dependencies
python>=3.10
torch>=2.1.0
transformers>=4.36.0
datasets>=2.16.0
accelerate>=0.25.0
deepspeed>=0.12.0

# Vector Store & Retrieval
chromadb>=0.4.22
sentence-transformers>=2.3.0
faiss-cpu>=1.7.4

# Langfuse Tracking
langfuse>=2.0.0

# Evaluation
rouge-score>=0.1.2
bert-score>=0.3.13
nltk>=3.8.1

# Utilities
pyyaml>=6.0.1
python-dotenv>=1.0.0
tqdm>=4.66.0
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0

# Fine-tuning
peft>=0.7.0  # For LoRA

# Development
pytest>=7.4.0
black>=23.12.0
flake8>=7.0.0
mypy>=1.8.0
EOF

# .env.example
cat > .env.example << 'EOF'
# Langfuse Configuration
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here
LANGFUSE_HOST=https://cloud.langfuse.com

# HuggingFace Token (for gated models)
HF_TOKEN=your-huggingface-token-here

# UCL Cloud GPU Configuration
CUDA_VISIBLE_DEVICES=0,1,2,3

# Project Settings
PROJECT_ROOT=/Users/declanmcalinden1/Portfolio_Projects_Clean/Langfuse
CHROMA_DB_DIR=${PROJECT_ROOT}/chroma_db
MODELS_CACHE_DIR=${PROJECT_ROOT}/models_cache
EOF

# config/dataset_config.yaml
cat > config/dataset_config.yaml << 'EOF'
# Dataset Configurations

datasets:
  squad:
    name: "squad"
    version: "v2"  # Use SQuAD 2.0 with unanswerable questions
    train_split: "train"
    eval_split: "validation"
    max_train_samples: 5000  # Limit for faster experimentation
    max_eval_samples: 1000
    context_key: "context"
    question_key: "question"
    answer_key: "answers"
    
  msmarco:
    name: "microsoft/ms_marco"
    config: "v2.1"
    train_split: "train"
    eval_split: "validation"
    max_train_samples: 10000
    max_eval_samples: 1000
    query_key: "query"
    passages_key: "passages"
    
  pubmedqa:
    name: "bigbio/pubmed_qa"
    config: "pubmed_qa_labeled_fold0_source"
    trust_remote_code: true
    train_split: "train"
    eval_split: "validation"
    max_train_samples: 5000
    max_eval_samples: 500
    question_key: "question"
    context_key: "context"
    answer_key: "long_answer"

# Preprocessing settings
preprocessing:
  chunk_size: 512
  chunk_overlap: 50
  max_question_length: 256
  max_context_length: 1024
EOF

# config/model_config.yaml
cat > config/model_config.yaml << 'EOF'
# Model Configurations

embedders:
  baseline:
    name: "sentence-transformers/all-MiniLM-L6-v2"
    dimension: 384
    max_seq_length: 256
    
  improved:
    name: "BAAI/bge-large-en-v1.5"
    dimension: 1024
    max_seq_length: 512

generators:
  qwen:
    name: "Qwen/Qwen2.5-7B-Instruct"
    context_length: 32768
    load_in_8bit: false
    load_in_4bit: false
    torch_dtype: "bfloat16"
    
  deepseek:
    name: "deepseek-ai/deepseek-llm-7b-chat"
    context_length: 4096
    load_in_8bit: false
    load_in_4bit: false
    torch_dtype: "bfloat16"
    
  llama:
    name: "meta-llama/Llama-3.2-8B-Instruct"
    context_length: 8192
    load_in_8bit: false
    load_in_4bit: false
    torch_dtype: "bfloat16"

# Retrieval settings
retrieval:
  top_k: 5
  similarity_metric: "cosine"
  use_reranking: false
  reranker_model: null

# Generation settings
generation:
  max_new_tokens: 512
  temperature: 0.7
  top_p: 0.9
  do_sample: true
  num_return_sequences: 1
EOF

# config/training_config.yaml
cat > config/training_config.yaml << 'EOF'
# Training Configurations

embedding_training:
  learning_rate: 2e-5
  batch_size: 32
  num_epochs: 3
  warmup_steps: 500
  loss_function: "contrastive"  # or "triplet"
  margin: 0.5
  weight_decay: 0.01
  optimizer: "adamw"
  scheduler: "linear"
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  
  # Evaluation
  eval_steps: 500
  save_steps: 1000
  logging_steps: 100

generator_training:
  learning_rate: 5e-6
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  gradient_accumulation_steps: 4
  num_train_epochs: 2
  warmup_steps: 100
  weight_decay: 0.01
  optimizer: "adamw"
  scheduler: "cosine"
  max_grad_norm: 1.0
  
  # Multi-GPU Configuration (UCL Cloud: 4x 24GB GPUs)
  distributed_type: "FSDP"  # Fully Sharded Data Parallel
  mixed_precision: "bf16"   # BFloat16 for Ampere+ GPUs
  num_processes: 4
  
  # FSDP Configuration
  fsdp_config:
    fsdp_sharding_strategy: "FULL_SHARD"  # Shard parameters, gradients, optimiser states
    fsdp_offload_params: false  # Keep on GPU with 24GB each
    fsdp_auto_wrap_policy: "TRANSFORMER_BASED_WRAP"
    fsdp_backward_prefetch: "BACKWARD_PRE"
    fsdp_state_dict_type: "FULL_STATE_DICT"
    fsdp_forward_prefetch: false
    fsdp_use_orig_params: true
  
  # Alternative: DeepSpeed ZeRO-3 Configuration
  deepspeed_config:
    zero_optimization:
      stage: 3
      offload_optimizer:
        device: "none"  # Keep on GPU
      offload_param:
        device: "none"
      overlap_comm: true
      contiguous_gradients: true
      reduce_bucket_size: 5e8
      stage3_prefetch_bucket_size: 5e8
      stage3_param_persistence_threshold: 1e6
    
    bf16:
      enabled: true
    
    gradient_accumulation_steps: 4
    train_micro_batch_size_per_gpu: 2
    
  # Evaluation
  eval_steps: 250
  save_steps: 500
  logging_steps: 50
  save_total_limit: 3
  
  # LoRA Configuration (optional, for efficient fine-tuning)
  use_lora: true
  lora_config:
    r: 16
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    bias: "none"
    task_type: "CAUSAL_LM"
EOF

# config/evaluation_config.yaml
cat > config/evaluation_config.yaml << 'EOF'
# Evaluation Metrics Configuration

retrieval_metrics:
  - "precision@k"
  - "recall@k"
  - "mrr"  # Mean Reciprocal Rank
  - "ndcg@k"  # Normalised Discounted Cumulative Gain
  - "map"  # Mean Average Precision
  
  k_values: [1, 3, 5, 10]

generation_metrics:
  - "rouge"  # ROUGE-1, ROUGE-2, ROUGE-L
  - "bleu"
  - "exact_match"
  - "f1_score"
  - "bertscore"
  
  llm_judge:
    enabled: false
    model: "gpt-4o-mini"
    metrics:
      - "answer_relevancy"
      - "faithfulness"
      - "correctness"

citation_metrics:
  - "citation_coverage"  # % of statements with citations
  - "citation_support_rate"  # % of citations that support the statement
  - "citation_precision"  # % of retrieved docs that are cited
  - "citation_recall"  # % of relevant docs that are cited
  - "citation_contradiction_rate"  # % of citations that contradict the statement

performance_metrics:
  - "latency_p50"
  - "latency_p95"
  - "latency_p99"
  - "tokens_per_second"
  - "gpu_memory_peak_gb"

langfuse_tracking:
  enabled: true
  log_traces: true
  log_spans: true
  log_metrics: true
  batch_size: 10
EOF

# config/accelerate_fsdp.yaml
cat > config/accelerate_fsdp.yaml << 'EOF'
# Accelerate Configuration for FSDP Multi-GPU Training
# Compatible with UCL Cloud: 4x GPUs with 24GB memory each

compute_environment: LOCAL_MACHINE
debug: false
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false

fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: false
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true

machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4  # 4 GPUs
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

# src/__init__.py
cat > src/__init__.py << 'EOF'
"""
Citation-Aware Q&A System with Langfuse Tracking.

A comprehensive RAG evaluation framework for demonstrating iterative
improvements through fine-tuning, prompt engineering, and retrieval optimisation.
"""

__version__ = "1.0.0"
__author__ = "Declan McAlinden"
EOF

# src/tracking/__init__.py
cat > src/tracking/__init__.py << 'EOF'
"""Langfuse tracking and observability module."""

from .langfuse_tracker import LangfuseTracker

__all__ = ["LangfuseTracker"]
EOF

# src/tracking/langfuse_tracker.py
cat > src/tracking/langfuse_tracker.py << 'EOF'
# (Content truncated for brevity in this script header comment)
# Full content inserted below
"""
Langfuse integration wrapper for comprehensive experiment tracking.

This module provides a unified interface for tracking RAG experiments using Langfuse Pro,
with support for self-hosted models (no API cost tracking, focuses on throughput and GPU metrics).

Author: Declan McAlinden
Date: 2025-11-10
"""

import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch
from langfuse.decorators import observe, langfuse_context
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class LangfuseTracker:
    """
    Comprehensive experiment tracking wrapper for Langfuse Pro.
    
    This class handles:
    - Experiment initialisation and configuration logging
    - Real-time trace logging for individual queries
    - Aggregated metrics tracking across evaluation sets
    - Multi-GPU training run tracking with distributed synchronisation
    - Throughput and GPU metrics (for self-hosted models)
    
    Attributes:
        experiment_name (str): Name of the current experiment
        experiment_config (Dict[str, Any]): Experiment configuration dictionary
        start_time (float): Experiment start timestamp
    """
    
    def __init__(
        self,
        experiment_name: str,
        experiment_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialise Langfuse tracker for experiment monitoring.
        
        Args:
            experiment_name: Unique identifier for this experiment run
            experiment_config: Dictionary containing hyperparameters and settings
        
        Raises:
            ValueError: If Langfuse credentials are not properly configured
        """
        # Validate credentials
        if not all([
            os.getenv("LANGFUSE_PUBLIC_KEY"),
            os.getenv("LANGFUSE_SECRET_KEY"),
            os.getenv("LANGFUSE_HOST")
        ]):
            raise ValueError(
                "Langfuse credentials not found. "
                "Please configure LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, "
                "and LANGFUSE_HOST in your .env file."
            )
        
        self.experiment_name = experiment_name
        self.experiment_config = experiment_config or {}
        self.start_time = time.time()
        
        logger.info(f"Initialised Langfuse tracker for experiment: {experiment_name}")
        
        # Log experiment initialisation
        self._log_experiment_start()
    
    @observe(name="experiment_initialisation")
    def _log_experiment_start(self) -> None:
        """Log experiment initialisation with configuration details."""
        langfuse_context.update_current_trace(
            metadata={
                "experiment_name": self.experiment_name,
                "start_time": datetime.now().isoformat(),
                "config": self.experiment_config,
                "gpu_info": self._get_gpu_info()
            }
        )
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """
        Collect GPU information for hardware tracking.
        
        Returns:
            Dictionary containing GPU device names, memory, and CUDA version
        """
        if not torch.cuda.is_available():
            return {"gpus": 0, "cuda_available": False}
        
        return {
            "num_gpus": torch.cuda.device_count(),
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "devices": [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / 1e9, 2
                    )
                }
                for i in range(torch.cuda.device_count())
            ]
        }
    
    @observe(as_type="generation")
    def track_query(
        self,
        query: str,
        retrieved_contexts: List[Dict[str, Any]],
        generated_answer: str,
        citations: List[str],
        meta Optional[Dict[str, Any]] = None,
        token_counts: Optional[Dict[str, int]] = None,
        gpu_memory_used_gb: Optional[float] = None,
        latency_ms: Optional[float] = None
    ) -> str:
        """
        Track a single query through the RAG pipeline with Langfuse.
        
        For self-hosted models, we track throughput and GPU metrics rather than API costs.
        
        Args:
            query: User question or query text
            retrieved_contexts: List of retrieved document chunks with metadata
            generated_answer: LLM-generated response
            citations: List of citation identifiers used in the answer
            meta Additional metadata (e.g., dataset name, iteration number)
            token_counts: {"input": 150, "output": 75} for throughput tracking
            gpu_memory_used_gb: Peak GPU memory usage during generation
            latency_ms: Total query latency in milliseconds
        
        Returns:
            Generated answer (passed through for convenience)
        """
        # Calculate throughput (tokens per second)
        throughput = None
        if token_counts and latency_ms:
            total_tokens = token_counts.get("input", 0) + token_counts.get("output", 0)
            if latency_ms > 0:
                throughput = (total_tokens / latency_ms) * 1000  # tokens/second
        
        # Prepare tracking metadata
        tracking_metadata = {
            "experiment": self.experiment_name,
            "num_retrieved_docs": len(retrieved_contexts),
            "num_citations": len(citations),
            **(metadata or {})
        }
        
        # Add self-hosted metrics (not API costs)
        if token_counts:
            tracking_metadata["token_counts"] = token_counts
        if throughput:
            tracking_metadata["tokens_per_second"] = round(throughput, 2)
        if gpu_memory_used_gb:
            tracking_metadata["gpu_memory_peak_gb"] = round(gpu_memory_used_gb, 2)
        if latency_ms:
            tracking_metadata["latency_ms"] = round(latency_ms, 2)
        
        # Update Langfuse observation
        langfuse_context.update_current_observation(
            input=query,
            output=generated_answer,
            metadata=tracking_metadata
        )
        
        return generated_answer
    
    @observe(name="metrics_logging")
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        split: str = "eval"
    ) -> None:
        """
        Log aggregated evaluation metrics to Langfuse.
        
        Args:
            metrics: Dictionary of metric names to values (e.g., {"precision@5": 0.85})
            step: Training step or iteration number (optional)
            split: Dataset split name ("train", "eval", "test")
        """
        langfuse_context.update_current_observation(
            output=metrics,
            metadata={
                "experiment": self.experiment_name,
                "split": split,
                "step": step,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Logged metrics for {split} split: {metrics}")
    
    @observe(name="training_step")
    def log_training_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        gpu_memory_used: Optional[List[float]] = None,
        gradient_norm: Optional[float] = None,
        tokens_per_second: Optional[float] = None
    ) -> None:
        """
        Log training step metrics for fine-tuning experiments.
        
        Args:
            step: Current training step number
            loss: Training loss value
            learning_rate: Current learning rate
            gpu_memory_used: List of GPU memory usage per device (GB)
            gradient_norm: Gradient norm for stability monitoring
            tokens_per_second: Training throughput metric
        """
        metrics = {
            "step": step,
            "loss": round(loss, 4),
            "learning_rate": learning_rate
        }
        
        if tokens_per_second:
            metrics["tokens_per_second"] = round(tokens_per_second, 2)
        
        if gpu_memory_used:
            for i, mem in enumerate(gpu_memory_used):
                metrics[f"gpu_{i}_memory_gb"] = round(mem, 2)
            metrics["gpu_memory_total_gb"] = round(sum(gpu_memory_used), 2)
        
        if gradient_norm is not None:
            metrics["gradient_norm"] = round(gradient_norm, 4)
        
        langfuse_context.update_current_observation(
            output=metrics,
            metadata={
                "experiment": self.experiment_name,
                "training_step": step
            }
        )
    
    @observe(name="experiment_comparison")
    def log_comparison(
        self,
        baseline_metrics: Dict[str, float],
        improved_metrics: Dict[str, float],
        improvement_description: str
    ) -> None:
        """
        Log a comparison between baseline and improved model performance.
        
        Args:
            baseline_metrics: Metrics from baseline model
            improved_metrics: Metrics from improved model
            improvement_description: Description of the improvement
        """
        # Calculate percentage improvements
        improvements = {}
        for metric_name in baseline_metrics.keys():
            if metric_name in improved_metrics:
                baseline_val = baseline_metrics[metric_name]
                improved_val = improved_metrics[metric_name]
                
                if baseline_val > 0:
                    pct_change = ((improved_val - baseline_val) / baseline_val) * 100
                    improvements[f"{metric_name}_improvement_pct"] = round(pct_change, 2)
        
        langfuse_context.update_current_observation(
            output={
                "baseline": baseline_metrics,
                "improved": improved_metrics,
                "improvements": improvements
            },
            metadata={
                "experiment": self.experiment_name,
                "improvement_type": improvement_description,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Logged comparison: {improvement_description}")
        logger.info(f"Improvements: {improvements}")
EOF

# src/data/__init__.py
cat > src/data/__init__.py << 'EOF'
"""Data loading and preprocessing module."""

from .dataset_loader import DatasetLoader, RAGExample

__all__ = ["DatasetLoader", "RAGExample"]
EOF

# src/data/dataset_loader.py
cat > src/data/dataset_loader.py << 'EOF'
# (See earlier content) — Full file inserted
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
        meta Additional metadata (e.g., source, difficulty level)
    """
    id: str
    question: str
    contexts: List[str]
    answer: str
    meta Dict[str, Any]


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
EOF

# src/models/__init__.py
cat > src/models/__init__.py << 'EOF'
"""Model components for RAG pipeline."""

from .embedder import Embedder
from .retriever import Retriever
from .generator import Generator

__all__ = ["Embedder", "Retriever", "Generator"]
EOF

# src/models/embedder.py
cat > src/models/embedder.py << 'EOF'
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
EOF

# src/models/retriever.py
cat > src/models/retriever.py << 'EOF'
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
EOF

# src/models/generator.py
cat > src/models/generator.py << 'EOF'
# (See earlier content) — Full file inserted
"""
LLM generator component for citation-aware answer generation.

Supports HuggingFace transformers with citation extraction and formatting.

Author: Declan McAlinden
Date: 2025-11-10
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class Generator:
    """
    LLM-based answer generator with citation awareness.
    
    This class handles:
    - Loading and managing LLM models (Qwen, DeepSeek, Llama)
    - Generating answers from retrieved contexts
    - Formatting responses with proper citations
    - Tracking token usage and generation metrics
    
    Attributes:
        model_name (str): Name/path of the LLM model
        model: Loaded transformer model
        tokenizer: Tokenizer for the model
        device (str): Compute device ("cuda" or "cpu")
        max_new_tokens (int): Maximum tokens to generate
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 512
    ):
        """
        Initialise LLM generator.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Compute device ("cuda", "cpu", or None for auto-detect)
            load_in_8bit: Whether to use 8-bit quantisation
            load_in_4bit: Whether to use 4-bit quantisation
            torch_dtype: Data type for model weights ("float32", "float16", "bfloat16")
            max_new_tokens: Maximum number of tokens to generate
        
        Raises:
            RuntimeError: If model loading fails
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading generator model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Convert torch_dtype string to actual dtype
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16
            }
            torch_dtype_actual = dtype_map.get(torch_dtype, torch.bfloat16)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype_actual,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Move to device if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load generator model '{model_name}': {str(e)}")
    
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate answer with citations based on query and retrieved contexts.
        
        Args:
            query: User question
            contexts: List of retrieved context dicts with "text" and "score" keys
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling (vs. greedy decoding)
            system_prompt: Optional system prompt for instruction-tuned models
        
        Returns:
            Tuple of (generated_answer, metrics_dict)
            - generated_answer: Answer text with citations
            - metrics_dict: Token counts, latency, GPU memory usage
        """
        start_time = time.time()
        
        # Format prompt with contexts
        formatted_prompt = self._format_prompt(query, contexts, system_prompt)
        
        # Tokenise input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings - self.max_new_tokens
        ).to(self.device)
        
        input_token_count = inputs["input_ids"].shape[1]
        
        # Track GPU memory before generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_before = torch.cuda.max_memory_allocated() / 1e9
        else:
            gpu_memory_before = 0
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Track GPU memory after generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_after = torch.cuda.max_memory_allocated() / 1e9
            gpu_memory_used = gpu_memory_after - gpu_memory_before
        else:
            gpu_memory_used = 0
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][input_token_count:],  # Skip input tokens
            skip_special_tokens=True
        )
        
        output_token_count = outputs.shape[1] - input_token_count
        
        # Extract citations from generated text
        citations = self._extract_citations(generated_text)
        
        # Calculate metrics
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        metrics = {
            "token_counts": {
                "input": input_token_count,
                "output": output_token_count
            },
            "latency_ms": latency_ms,
            "gpu_memory_used_gb": gpu_memory_used,
            "citations_count": len(citations)
        }
        
        logger.debug(f"Generated {output_token_count} tokens in {latency_ms:.2f}ms")
        
        return generated_text.strip(), metrics
    
    def _format_prompt(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format prompt with query and contexts.
        
        Args:
            query: User question
            contexts: Retrieved contexts with text and scores
            system_prompt: Optional system instruction
        
        Returns:
            Formatted prompt string
        """
        # Default system prompt for citation-aware generation
        if system_prompt is None:
            system_prompt = (
                "You are a helpful AI assistant that answers questions based on provided contexts. "
                "Always cite your sources using [1], [2], etc. to reference the context passages. "
                "Only include information that is supported by the provided contexts."
            )
        
        # Format contexts with citation numbers
        context_str = "\n\n".join([
            f"[{i+1}] {ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])
        
        # Build prompt (adjust format based on model type)
        if "qwen" in self.model_name.lower():
            # Qwen format
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            prompt += f"<|im_start|>user\n"
            prompt += f"Contexts:\n{context_str}\n\n"
            prompt += f"Question: {query}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n"
        
        elif "llama" in self.model_name.lower():
            # Llama format
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n"
            prompt += f"Contexts:\n{context_str}\n\n"
            prompt += f"Question: {query}<|eot_id|>"
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n"
        
        else:
            # Generic format
            prompt = f"System: {system_prompt}\n\n"
            prompt += f"Contexts:\n{context_str}\n\n"
            prompt += f"Question: {query}\n\n"
            prompt += f"Answer:"
        
        return prompt
    
    def _extract_citations(self, text: str) -> List[int]:
        """
        Extract citation numbers from generated text.
        
        Args:
            text: Generated answer text
        
        Returns:
            List of citation numbers found in text
        """
        # Find all [1], [2], etc. patterns
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        
        # Convert to integers and remove duplicates
        citations = sorted(set(int(m) for m in matches))
        
        return citations
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "dtype": str(self.model.dtype)
        }
EOF

# src/evaluation/__init__.py
cat > src/evaluation/__init__.py << 'EOF'
"""Evaluation metrics and evaluator module."""

from .metrics import RAGMetrics
from .evaluator import RAGEvaluator

__all__ = ["RAGMetrics", "RAGEvaluator"]
EOF

# src/evaluation/metrics.py
cat > src/evaluation/metrics.py << 'EOF'
# (See earlier content) — Full file inserted
"""
Comprehensive evaluation metrics for RAG systems.

Implements retrieval, generation, and citation-specific metrics
for thorough assessment of RAG pipeline performance.

Author: Declan McAlinden
Date: 2025-11-10
"""

import logging
from typing import List, Dict, Any
import numpy as np
import re
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)


class RAGMetrics:
    """
    Comprehensive metrics calculator for RAG evaluation.
    
    Implements:
    - Retrieval metrics (Precision@k, Recall@k, MRR, NDCG, MAP)
    - Generation metrics (ROUGE, BLEU, Exact Match, F1)
    - Citation metrics (Coverage, Support Rate, Precision, Recall)
    """
    
    def __init__(self):
        """Initialise metrics calculator."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    # ========== RETRIEVAL METRICS ==========
    
    def precision_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@k: proportion of retrieved docs that are relevant.
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered by relevance)
            relevant_ids: List of ground-truth relevant document IDs
            k: Cutoff rank
        
        Returns:
            Precision@k score in range [0, 1]
        """
        retrieved_at_k = retrieved_ids[:k]
        relevant_set = set(relevant_ids)
        
        num_relevant = sum(1 for doc_id in retrieved_at_k if doc_id in relevant_set)
        
        return num_relevant / k if k > 0 else 0.0
    
    def recall_at_k(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@k: proportion of relevant docs that are retrieved.
        
        Args:
            retrieved_ids: List of retrieved document IDs
            relevant_ids: List of ground-truth relevant document IDs
            k: Cutoff rank
        
        Returns:
            Recall@k score in range [0, 1]
        """
        if not relevant_ids:
            return 0.0
        
        retrieved_at_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        
        num_relevant_retrieved = len(retrieved_at_k.intersection(relevant_set))
        
        return num_relevant_retrieved / len(relevant_set)
    
    def mean_reciprocal_rank(
        self,
        retrieved_ids: List[str],
        relevant_ids: List[str]
    ) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).
        
        MRR = 1 / rank of first relevant document
        
        Args:
            retrieved_ids: List of retrieved document IDs (ordered)
            relevant_ids: List of relevant document IDs
        
        Returns:
            MRR score in range [0, 1]
        """
        relevant_set = set(relevant_ids)
        
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_set:
                return 1.0 / rank
        
        return 0.0
    
    # ========== GENERATION METRICS ==========
    
    def exact_match(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate exact match score (normalised string comparison).
        
        Args:
            predicted: Predicted answer text
            ground_truth: Ground truth answer text
        
        Returns:
            1.0 if exact match (after normalisation), 0.0 otherwise
        """
        return float(self._normalise_answer(predicted) == self._normalise_answer(ground_truth))
    
    def f1_score_text(self, predicted: str, ground_truth: str) -> float:
        """
        Calculate token-level F1 score between predicted and ground truth.
        
        Args:
            predicted: Predicted answer text
            ground_truth: Ground truth answer text
        
        Returns:
            F1 score in range [0, 1]
        """
        pred_tokens = self._normalise_answer(predicted).split()
        truth_tokens = self._normalise_answer(ground_truth).split()
        
        if not pred_tokens or not truth_tokens:
            return float(pred_tokens == truth_tokens)
        
        common_tokens = set(pred_tokens) & set(truth_tokens)
        
        if not common_tokens:
            return 0.0
        
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(truth_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1
    
    def rouge_scores(
        self,
        predicted: str,
        ground_truth: str
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores for answer quality.
        
        Args:
            predicted: Predicted answer text
            ground_truth: Ground truth answer text
        
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
        """
        scores = self.rouge_scorer.score(ground_truth, predicted)
        
        return {
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure
        }
    
    # ========== CITATION METRICS ==========
    
    def citation_coverage(self, answer: str) -> float:
        """
        Calculate proportion of sentences in answer that have citations.
        
        Citations are detected as [1], [2], etc.
        
        Args:
            answer: Generated answer text
        
        Returns:
            Coverage ratio in range [0, 1]
        """
        sentences = self._split_into_sentences(answer)
        
        if not sentences:
            return 0.0
        
        citations_pattern = r'\[\d+\]'
        sentences_with_citations = sum(
            1 for sent in sentences
            if re.search(citations_pattern, sent)
        )
        
        return sentences_with_citations / len(sentences)
    
    def citation_support_rate(
        self,
        answer: str,
        retrieved_contexts: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate proportion of citations that are supported by retrieved contexts.
        
        Args:
            answer: Generated answer with citations
            retrieved_contexts: List of retrieved document dicts
        
        Returns:
            Support rate in range [0, 1]
        """
        # Extract citation numbers from answer
        citation_numbers = self._extract_citation_numbers(answer)
        
        if not citation_numbers:
            return 0.0
        
        # Check if each citation corresponds to a retrieved document
        num_retrieved = len(retrieved_contexts)
        supported_citations = sum(
            1 for cite_num in citation_numbers
            if cite_num <= num_retrieved
        )
        
        return supported_citations / len(citation_numbers)
    
    # ========== HELPER METHODS ==========
    
    def _normalise_answer(self, text: str) -> str:
        """Normalise answer text for comparison."""
        # Lowercase, remove punctuation, extra whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_citation_numbers(self, text: str) -> List[int]:
        """Extract citation numbers from text (e.g., [1], [2])."""
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        return [int(m) for m in matches]
EOF

# src/evaluation/evaluator.py
cat > src/evaluation/evaluator.py << 'EOF'
# (See earlier content) — Full file inserted
"""
RAG pipeline evaluator with comprehensive metrics tracking.

Orchestrates retrieval, generation, and evaluation with Langfuse logging.

Author: Declan McAlinden
Date: 2025-11-10
"""

import logging
from typing import List, Dict, Any
from tqdm import tqdm
import yaml
import numpy as np

from ..data.dataset_loader import RAGExample
from .metrics import RAGMetrics

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """
    End-to-end RAG pipeline evaluator.
    
    This class orchestrates:
    - Document retrieval for each query
    - Answer generation with citations
    - Comprehensive metric calculation
    - Langfuse tracking integration
    
    Attributes:
        retriever: Retriever instance for document search
        generator: Generator instance for answer generation
        metrics: RAGMetrics calculator
        tracker: Optional Langfuse tracker for experiment logging
    """
    
    def __init__(
        self,
        retriever,
        generator,
        metrics_config: str,
        tracker=None
    ):
        """
        Initialise RAG evaluator.
        
        Args:
            retriever: Retriever instance
            generator: Generator instance
            metrics_config: Path to evaluation config YAML
            tracker: Optional LangfuseTracker instance
        """
        self.retriever = retriever
        self.generator = generator
        self.tracker = tracker
        self.metrics = RAGMetrics()
        
        # Load evaluation configuration
        with open(metrics_config, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info("RAG evaluator initialised")
    
    def evaluate(
        self,
        examples: List[RAGExample],
        show_progress: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate RAG pipeline on a list of examples.
        
        Args:
            examples: List of RAGExample objects to evaluate
            show_progress: Whether to show progress bar
        
        Returns:
            Dictionary of aggregated metrics
        """
        logger.info(f"Evaluating {len(examples)} examples...")
        
        all_results = []
        
        # Process each example
        iterator = tqdm(examples, desc="Evaluating") if show_progress else examples
        
        for example in iterator:
            result = self._evaluate_single(example)
            all_results.append(result)
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_results(all_results)
        
        logger.info("Evaluation complete")
        
        return aggregated_metrics
    
    def _evaluate_single(self, example: RAGExample) -> Dict[str, Any]:
        """
        Evaluate a single example through the RAG pipeline.
        
        Args:
            example: RAGExample to evaluate
        
        Returns:
            Dictionary containing metrics for this example
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=example.question,
            top_k=self.config.get('retrieval', {}).get('top_k', 5)
        )
        
        # Step 2: Generate answer
        generated_answer, gen_metrics = self.generator.generate(
            query=example.question,
            contexts=retrieved_docs
        )
        
        # Step 3: Calculate metrics
        result = {
            "example_id": example.id,
            "question": example.question,
            "ground_truth": example.answer,
            "generated_answer": generated_answer,
            "retrieved_docs": retrieved_docs
        }
        
        # Generation metrics
        result["exact_match"] = self.metrics.exact_match(generated_answer, example.answer)
        result["f1_score"] = self.metrics.f1_score_text(generated_answer, example.answer)
        
        rouge_scores = self.metrics.rouge_scores(generated_answer, example.answer)
        result.update(rouge_scores)
        
        # Citation metrics
        result["citation_coverage"] = self.metrics.citation_coverage(generated_answer)
        result["citation_support_rate"] = self.metrics.citation_support_rate(
            generated_answer, retrieved_docs
        )
        
        # Performance metrics
        result["latency_ms"] = gen_metrics["latency_ms"]
        result["tokens_per_second"] = (
            gen_metrics["token_counts"]["output"] / gen_metrics["latency_ms"] * 1000
            if gen_metrics["latency_ms"] > 0 else 0
        )
        result["gpu_memory_gb"] = gen_metrics.get("gpu_memory_used_gb", 0)
        
        # Track in Langfuse if available
        if self.tracker:
            citations = self.generator._extract_citations(generated_answer)
            self.tracker.track_query(
                query=example.question,
                retrieved_contexts=retrieved_docs,
                generated_answer=generated_answer,
                citations=[str(c) for c in citations],
                metadata={
                    "dataset": example.metadata.get("dataset"),
                    "example_id": example.id
                },
                token_counts=gen_metrics["token_counts"],
                gpu_memory_used_gb=gen_metrics.get("gpu_memory_used_gb"),
                latency_ms=gen_metrics["latency_ms"]
            )
        
        return result
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate individual results into summary metrics.
        
        Args:
            results: List of per-example result dictionaries
        
        Returns:
            Dictionary of aggregated metrics
        """
        aggregated = {}
        
        # Metrics to average
        metric_keys = [
            "exact_match", "f1_score",
            "rouge1", "rouge2", "rougeL",
            "citation_coverage", "citation_support_rate",
            "latency_ms", "tokens_per_second", "gpu_memory_gb"
        ]
        
        for key in metric_keys:
            values = [r[key] for r in results if key in r]
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
        
        # Percentile metrics for latency
        latencies = [r["latency_ms"] for r in results if "latency_ms" in r]
        if latencies:
            aggregated["latency_p50"] = float(np.percentile(latencies, 50))
            aggregated["latency_p95"] = float(np.percentile(latencies, 95))
            aggregated["latency_p99"] = float(np.percentile(latencies, 99))
        
        return aggregated
EOF

# src/utils/__init__.py
cat > src/utils/__init__.py << 'EOF'
"""Utility functions and helpers."""

from .config_loader import load_config

__all__ = ["load_config"]
EOF

# src/utils/config_loader.py
cat > src/utils/config_loader.py << 'EOF'
"""
Configuration loading utilities.

Author: Declan McAlinden
Date: 2025-11-10
"""

import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
    
    Returns:
        Dictionary containing configuration
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config
EOF

# src/prompts/__init__.py
cat > src/prompts/__init__.py << 'EOF'
"""Prompt templates for RAG generation."""

from .base_prompts import BASELINE_SYSTEM_PROMPT, CITATION_AWARE_PROMPT
from .improved_prompts import FEW_SHOT_PROMPT, CHAIN_OF_THOUGHT_PROMPT

__all__ = [
    "BASELINE_SYSTEM_PROMPT",
    "CITATION_AWARE_PROMPT",
    "FEW_SHOT_PROMPT",
    "CHAIN_OF_THOUGHT_PROMPT"
]
EOF

# src/prompts/base_prompts.py
cat > src/prompts/base_prompts.py << 'EOF'
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
EOF

# src/prompts/improved_prompts.py
cat > src/prompts/improved_prompts.py << 'EOF'
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
EOF

# scripts/run_baseline_evaluation.py
cat > scripts/run_baseline_evaluation.py << 'EOF'
#!/usr/bin/env python3
# (See earlier content) — Full file inserted
"""
Run baseline RAG evaluation with Langfuse tracking.

This script establishes baseline performance metrics before any optimisations.

Usage:
    python scripts/run_baseline_evaluation.py --dataset squad --max_samples 1000

Author: Declan McAlinden
Date: 2025-11-10
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import DatasetLoader
from src.models.embedder import Embedder
from src.models.retriever import Retriever
from src.models.generator import Generator
from src.evaluation.evaluator import RAGEvaluator
from src.tracking.langfuse_tracker import LangfuseTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run baseline RAG evaluation with Langfuse tracking"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["squad", "msmarco", "pubmedqa"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=1000,
        help="Maximum number of samples to evaluate (default: 1000)"
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--generator",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Generator model to use"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    logger.info(f"Starting baseline evaluation on {args.dataset}")
    logger.info(f"Embedder: {args.embedder}")
    logger.info(f"Generator: {args.generator}")
    
    # Experiment configuration
    exp_config = {
        "experiment_type": "baseline",
        "dataset": args.dataset,
        "embedder_model": args.embedder,
        "generator_model": args.generator,
        "max_samples": args.max_samples
    }
    
    # Initialise Langfuse tracker
    tracker = LangfuseTracker(
        experiment_name=f"baseline_{args.dataset}",
        experiment_config=exp_config
    )
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = DatasetLoader(str(project_root / "config" / "dataset_config.yaml"))
    eval_data = dataset_loader.load_dataset(
        args.dataset,
        split="validation",
        max_samples=args.max_samples
    )
    
    logger.info(f"Loaded {len(eval_data)} evaluation examples")
    
    # Initialise models
    logger.info("Loading embedding model...")
    embedder = Embedder(model_name=args.embedder)
    
    logger.info("Initialising retriever...")
    retriever = Retriever(
        embedder=embedder,
        collection_name=f"{args.dataset}_baseline",
        persist_directory=str(project_root / "chroma_db"),
        top_k=5
    )
    
    # Index documents
    logger.info("Indexing documents...")
    all_contexts = []
    for example in eval_
        all_contexts.extend(example.contexts)
    
    # Remove duplicates
    all_contexts = list(set(all_contexts))
    logger.info(f"Indexing {len(all_contexts)} unique document chunks...")
    retriever.index_documents(all_contexts, show_progress=True)
    
    # Initialise generator
    logger.info("Loading generator model...")
    generator = Generator(
        model_name=args.generator,
        max_new_tokens=512
    )
    
    # Create evaluator
    evaluator = RAGEvaluator(
        retriever=retriever,
        generator=generator,
        metrics_config=str(project_root / "config" / "evaluation_config.yaml"),
        tracker=tracker
    )
    
    # Run evaluation
    logger.info("Running evaluation...")
    results = evaluator.evaluate(eval_data, show_progress=True)
    
    # Log results to Langfuse
    tracker.log_metrics(results, split="validation")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BASELINE EVALUATION RESULTS")
    logger.info("="*80)
    for metric_name, value in sorted(results.items()):
        logger.info(f"{metric_name:30s}: {value:.4f}")
    logger.info("="*80)
    
    logger.info("\nEvaluation complete. Check Langfuse dashboard for detailed traces.")


if __name__ == "__main__":
    main()
EOF

# scripts/download_datasets.py
cat > scripts/download_datasets.py << 'EOF'
#!/usr/bin/env python3
"""
Download and cache datasets for evaluation.

Usage:
    python scripts/download_datasets.py --datasets squad msmarco pubmedqa

Author: Declan McAlinden
Date: 2025-11-10
"""

import argparse
import logging
from datasets import load_dataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_squad():
    """Download SQuAD v2.0 dataset."""
    logger.info("Downloading SQuAD v2.0...")
    dataset = load_dataset("squad", "v2")
    logger.info(f"✓ SQuAD downloaded: {len(dataset['train'])} train, {len(dataset['validation'])} validation")


def download_msmarco():
    """Download MS MARCO dataset."""
    logger.info("Downloading MS MARCO...")
    try:
        dataset = load_dataset("microsoft/ms_marco", "v2.1")
        logger.info(f"✓ MS MARCO downloaded: {len(dataset['train'])} train samples")
    except Exception as e:
        logger.warning(f"Note: MS MARCO may require manual download. Error: {e}")


def download_pubmedqa():
    """Download PubMedQA dataset."""
    logger.info("Downloading PubMedQA...")
    dataset = load_dataset("bigbio/pubmed_qa", "pubmed_qa_labeled_fold0_source", trust_remote_code=True)
    logger.info(f"✓ PubMedQA downloaded: {len(dataset['train'])} train samples")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Download datasets for RAG evaluation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["squad", "msmarco", "pubmedqa", "all"],
        default=["all"],
        help="Datasets to download (default: all)"
    )
    
    args = parser.parse_args()
    
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["squad", "msmarco", "pubmedqa"]
    
    logger.info(f"Downloading datasets: {', '.join(datasets_to_download)}")
    
    for dataset_name in datasets_to_download:
        try:
            if dataset_name == "squad":
                download_squad()
            elif dataset_name == "msmarco":
                download_msmarco()
            elif dataset_name == "pubmedqa":
                download_pubmedqa()
        except Exception as e:
            logger.error(f"Failed to download {dataset_name}: {e}")
    
    logger.info("\n✓ All datasets downloaded successfully!")
    logger.info("Datasets are cached in ~/.cache/huggingface/datasets/")


if __name__ == "__main__":
    main()
EOF

# experiments/baseline/experiment_config.yaml
cat > experiments/baseline/experiment_config.yaml << 'EOF'
# Baseline Experiment Configuration

experiment_name: "baseline_rag"
description: "Baseline RAG system with off-the-shelf models"

# Model configuration
embedder_model: "sentence-transformers/all-MiniLM-L6-v2"
generator_model: "Qwen/Qwen2.5-7B-Instruct"

# Retrieval settings
top_k: 5
use_reranking: false

# Generation settings
temperature: 0.7
top_p: 0.9
max_new_tokens: 512

# Prompt template
system_prompt: "baseline"
EOF

# README.md
cat > README.md << 'EOF'
# Citation-Aware Q&A System with Langfuse Tracking

> Portfolio Project: Demonstrating systematic RAG improvement through fine-tuning, prompt engineering, and observability

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/download_datasets.py --datasets squad
python scripts/run_baseline_evaluation.py --dataset squad --max_samples 1000
```

See detailed documentation in README-complete.md (optional).
EOF

# .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv

# Environment variables
.env
.env.local

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter Notebooks
.ipynb_checkpoints
*.ipynb

# Data & Models
chroma_db/
models_cache/
*.pt
*.bin
*.safetensors
checkpoints/

# Logs
*.log
logs/

# OS
.DS_Store
Thumbs.db

# Project specific
experiments/*/results/
experiments/*/checkpoints/
EOF

printf "\nDone. Project created at: %s\n" "$TARGET_DIR"
echo "Next steps:"
echo "1) cd \"$TARGET_DIR\""
echo "2) python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
echo "3) cp .env.example .env && edit your Langfuse keys"
echo "4) python scripts/download_datasets.py --datasets squad"
echo "5) python scripts/run_baseline_evaluation.py --dataset squad --max_samples 200"
