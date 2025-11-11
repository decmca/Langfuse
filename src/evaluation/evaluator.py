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
        show_progress: bool = True,
        batch_size: int = 8
    ) -> Dict[str, float]:
        """
        Evaluate RAG pipeline on a list of examples with batched processing.
        
        Args:
            examples: List of RAGExample objects to evaluate
            show_progress: Whether to show progress bar
            batch_size: Number of examples to process in each batch
        
        Returns:
            Dictionary of aggregated metrics
        """
        logger.info(f"Evaluating {len(examples)} examples with batch_size={batch_size}...")
        
        all_results = []
        
        # Process examples in batches
        num_batches = (len(examples) + batch_size - 1) // batch_size
        
        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Evaluating batches", total=num_batches)
        
        for batch_idx in iterator:
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(examples))
            batch_examples = examples[start_idx:end_idx]
            
            # Process batch
            batch_results = self._evaluate_batch(batch_examples)
            all_results.extend(batch_results)
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_results(all_results)
        
        logger.info("Evaluation complete")
        
        return aggregated_metrics
    
    def _evaluate_batch(self, examples: List[RAGExample]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of examples through the RAG pipeline.
        
        Args:
            examples: List of RAGExample objects to evaluate in batch
        
        Returns:
            List of dictionaries containing metrics for each example
        """
        batch_size = len(examples)
        
        # Step 1: Retrieve relevant documents for all queries
        all_retrieved_docs = []
        for example in examples:
            retrieved_docs = self.retriever.retrieve(
                query=example.question,
                top_k=self.config.get('retrieval', {}).get('top_k', 5)
            )
            all_retrieved_docs.append(retrieved_docs)
        
        # Step 2: Batch generate answers
        queries = [ex.question for ex in examples]
        
        generated_results = self.generator.generate_batch(
            queries=queries,
            contexts_list=all_retrieved_docs,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        # Step 3: Calculate metrics for each example
        results = []
        for i, example in enumerate(examples):
            generated_answer, gen_metrics = generated_results[i]
            retrieved_docs = all_retrieved_docs[i]
            
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
            result["batch_latency_ms"] = gen_metrics.get("batch_latency_ms", 0)
            result["batch_size_used"] = gen_metrics.get("batch_size", 1)
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
                        "example_id": example.id,
                        "batch_size": gen_metrics.get("batch_size", 1)
                    },
                    token_counts=gen_metrics["token_counts"],
                    gpu_memory_used_gb=gen_metrics.get("gpu_memory_used_gb"),
                    latency_ms=gen_metrics["latency_ms"]
                )
            
            results.append(result)
        
        return results
    
    def _evaluate_single(self, example: RAGExample) -> Dict[str, Any]:
        """
        [DEPRECATED] Evaluate a single example through the RAG pipeline.
        
        Use _evaluate_batch() with batch_size=1 instead for consistency.
        This method is kept for backwards compatibility only.
        
        Args:
            example: RAGExample to evaluate
        
        Returns:
            Dictionary containing metrics for this example
        """
        # Delegate to batch method with single example
        return self._evaluate_batch([example])[0]

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
