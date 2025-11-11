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
        # Access top_k from the new nested retrieval config
        top_k = self.config.get('retrieval', {}).get('top_k', 5)
        
        retrieved_docs = self.retriever.retrieve(
            query=example.question,
            top_k=top_k
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
