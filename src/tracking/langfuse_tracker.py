"""
Langfuse integration wrapper for comprehensive experiment tracking.

This module provides a unified interface for tracking RAG experiments using Langfuse Pro,
with support for self-hosted models (no API cost tracking, focuses on throughput and GPU metrics).

Author: Declan McAlinden
Date: 2025-11-10
Updated: 2025-11-11 (Langfuse v3 compatibility)
"""

import os
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch
from langfuse import observe, Langfuse
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
        langfuse (Langfuse): Langfuse client instance
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
        
        # Initialize Langfuse client
        self.langfuse = Langfuse()
        
        logger.info(f"Initialised Langfuse tracker for experiment: {experiment_name}")
        
        # Log experiment initialisation
        self._log_experiment_start()
    
    @observe(name="experiment_initialisation")
    def _log_experiment_start(self) -> None:
        """Log experiment initialisation with configuration details."""
        self.langfuse.update_current_trace(
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
        metadata: Optional[Dict[str, Any]] = None,
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
            metadata: Additional metadata (e.g., dataset name, iteration number)
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
        
        # Update Langfuse observation (v3 syntax)
        self.langfuse.update_current_generation(
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
        self.langfuse.update_current_span(
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
        
        self.langfuse.update_current_span(
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
        
        self.langfuse.update_current_span(
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
