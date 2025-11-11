#!/usr/bin/env python3
"""
Verify LanceDB installation and functionality.

Tests basic LanceDB operations to ensure migration was successful.

Author: Declan McAlinden
Date: 2025-11-11
"""

import sys
from pathlib import Path
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_lancedb_import():
    """Test LanceDB can be imported."""
    try:
        import lancedb
        import pyarrow
        logger.info("✓ LanceDB and PyArrow imports successful")
        logger.info(f"  LanceDB version: {lancedb.__version__}")
        logger.info(f"  PyArrow version: {pyarrow.__version__}")
        return True
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        return False


def test_embedder():
    """Test Embedder class."""
    try:
        from src.models.embedder import Embedder
        
        embedder = Embedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        
        # Test encoding
        texts = ["Hello world", "LanceDB is fast"]
        embeddings = embedder.encode(texts, show_progress_bar=False)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == embedder.dimension
        
        logger.info(f"✓ Embedder working (dimension: {embedder.dimension})")
        return True
    except Exception as e:
        logger.error(f"✗ Embedder error: {e}")
        return False


def test_retriever():
    """Test LanceRetriever class."""
    try:
        from src.models.embedder import Embedder
        from src.models.retriever_lance import LanceRetriever
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = Embedder(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
            
            retriever = LanceRetriever(
                embedder=embedder,
                table_name="test_table",
                persist_directory=tmpdir,
                top_k=2
            )
            
            # Index test documents
            docs = [
                "The quick brown fox jumps over the lazy dog",
                "Machine learning is a subset of artificial intelligence",
                "Python is a popular programming language"
            ]
            
            retriever.index_documents(docs, show_progress=False)
            
            # Test retrieval
            results = retriever.retrieve("What is Python?")
            
            assert len(results) <= 2
            assert all("text" in r for r in results)
            assert all("score" in r for r in results)
            
            logger.info(f"✓ LanceRetriever working (retrieved {len(results)} docs)")
            logger.info(f"  Top result score: {results[0]['score']:.4f}")
            
            # Test stats
            stats = retriever.get_stats()
            logger.info(f"  Stats: {stats}")
            
            return True
    except Exception as e:
        logger.error(f"✗ Retriever error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_factory():
    """Test retriever factory."""
    try:
        from src.models.embedder import Embedder
        from src.models.retriever_factory import create_retriever
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedder = Embedder(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cpu"
            )
            
            # Test with explicit path
            retriever = create_retriever(
                embedder=embedder,
                collection_name="test_collection",
                persist_directory=tmpdir
            )
            
            logger.info("✓ Retriever factory working")
            return True
    except Exception as e:
        logger.error(f"✗ Factory error: {e}")
        return False


def main():
    logger.info("="*80)
    logger.info("LanceDB VERIFICATION")
    logger.info("="*80 + "\n")
    
    tests = [
        ("Import Test", test_lancedb_import),
        ("Embedder Test", test_embedder),
        ("Retriever Test", test_retriever),
        ("Factory Test", test_factory)
    ]
    
    results = []
    for name, test_func in tests:
        logger.info(f"\nRunning: {name}")
        logger.info("-" * 40)
        success = test_func()
        results.append((name, success))
    
    logger.info("\n" + "="*80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*80)
    
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{name:20s}: {status}")
    
    all_passed = all(success for _, success in results)
    
    logger.info("="*80)
    
    if all_passed:
        logger.info("\n✓ All tests passed! LanceDB is ready to use.\n")
        return 0
    else:
        logger.error("\n✗ Some tests failed. Please check the errors above.\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
