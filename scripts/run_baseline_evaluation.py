#!/usr/bin/env python3
"""
Run baseline RAG evaluation with Langfuse tracking.

This script establishes baseline performance metrics before any optimisations.
Uses LanceDB for vector storage.

Usage:
    python scripts/run_baseline_evaluation.py --dataset squad --max_samples 1000

Author: Declan McAlinden
Date: 2025-11-11
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
from src.models.retriever_factory import create_retriever
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
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8, increase for better GPU utilisation)"
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
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of documents to retrieve per query (default: 5)"
    )
    parser.add_argument(
        "--distance_type",
        type=str,
        default="cosine",
        choices=["cosine", "l2", "dot"],
        help="Distance metric for LanceDB (default: cosine)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    logger.info("="*80)
    logger.info("BASELINE RAG EVALUATION - LanceDB Backend")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Embedder: {args.embedder}")
    logger.info(f"Generator: {args.generator}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Top-k retrieval: {args.top_k}")
    logger.info(f"Distance metric: {args.distance_type}")
    logger.info("="*80 + "\n")
    
    # Experiment configuration
    exp_config = {
        "experiment_type": "baseline",
        "dataset": args.dataset,
        "embedder_model": args.embedder,
        "generator_model": args.generator,
        "max_samples": args.max_samples,
        "batch_size": args.batch_size,
        "vector_store": "lancedb",
        "top_k": args.top_k,
        "distance_type": args.distance_type
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
    
    logger.info(f"Loaded {len(eval_data)} evaluation examples\n")
    
    # Initialise models
    logger.info("Loading embedding model...")
    embedder = Embedder(model_name=args.embedder)
    
    logger.info("Initialising LanceDB retriever...")
    retriever = create_retriever(
        embedder=embedder,
        collection_name=f"{args.dataset}_baseline",
        top_k=args.top_k,
        distance_type=args.distance_type
    )
    
    # Index documents
    logger.info("Preparing document corpus...")
    all_contexts = []
    for example in eval_data:
        all_contexts.extend(example.contexts)
    
    # Remove duplicates whilst preserving order
    seen = set()
    unique_contexts = []
    for ctx in all_contexts:
        if ctx not in seen:
            seen.add(ctx)
            unique_contexts.append(ctx)
    
    logger.info(f"Indexing {len(unique_contexts)} unique document chunks...")
    retriever.index_documents(unique_contexts, show_progress=True)
    
    # Verify indexing
    stats = retriever.get_stats()
    logger.info(f"Index statistics: {stats}\n")
    
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
    logger.info("="*80)
    results = evaluator.evaluate(
        eval_data, 
        show_progress=True,
        batch_size=args.batch_size
    )
    
    # Log results to Langfuse
    tracker.log_metrics(results, split="validation")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("BASELINE EVALUATION RESULTS")
    logger.info("="*80)
    for metric_name, value in sorted(results.items()):
        logger.info(f"{metric_name:30s}: {value:.4f}")
    logger.info("="*80)
    
    logger.info("\n✓ Evaluation complete. Check Langfuse dashboard for detailed traces.")
    logger.info(f"✓ Vector database persisted at: {retriever.persist_directory}")


if __name__ == "__main__":
    main()
