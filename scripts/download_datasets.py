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
