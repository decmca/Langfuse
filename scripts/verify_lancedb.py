#!/usr/bin/env python3
"""
Quick verification script for LanceDB installation and functionality.

This script performs basic tests to ensure LanceDB is properly integrated.

Author: Declan McAlinden
Date: 2025-11-11
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("LanceDB Integration Verification")
print("=" * 60)

# Test 1: Import check
print("\n1. Checking imports...")
try:
    import lancedb
    import pyarrow
    from src.models import Embedder, LanceRetriever, create_retriever
    print(f"   ✓ LanceDB version: {lancedb.__version__}")
    print(f"   ✓ PyArrow version: {pyarrow.__version__}")
    print("   ✓ All imports successful")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create retriever
print("\n2. Testing LanceDB retriever creation...")
try:
    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = LanceRetriever(
        embedder=embedder,
        table_name="verification_test",
        persist_directory="./lance_db_test"
    )
    print("   ✓ Retriever created successfully")
except Exception as e:
    print(f"   ✗ Retriever creation failed: {e}")
    sys.exit(1)

# Test 3: Index documents
print("\n3. Testing document indexing...")
try:
    test_docs = [
        "LanceDB is a high-performance vector database for AI applications.",
        "RAG systems retrieve relevant documents before generation.",
        "Python is widely used in machine learning projects."
    ]
    
    retriever.index_documents(test_docs, show_progress=False)
    stats = retriever.get_stats()
    print(f"   ✓ Indexed {stats['count']} documents")
except Exception as e:
    print(f"   ✗ Indexing failed: {e}")
    sys.exit(1)

# Test 4: Test retrieval
print("\n4. Testing document retrieval...")
try:
    results = retriever.retrieve("What is LanceDB?", top_k=2)
    
    if len(results) > 0:
        print(f"   ✓ Retrieved {len(results)} documents")
        print(f"\n   Top result (score: {results[0]['score']:.3f}):")
        print(f"   '{results[0]['text'][:70]}...'")
    else:
        print("   ✗ No documents retrieved")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Retrieval failed: {e}")
    sys.exit(1)

# Test 5: Test factory pattern
print("\n5. Testing factory pattern...")
try:
    import os
    os.environ["VECTOR_STORE_TYPE"] = "lancedb"
    
    factory_retriever = create_retriever(
        embedder=embedder,
        collection_name="factory_test"
    )
    
    if isinstance(factory_retriever, LanceRetriever):
        print("   ✓ Factory correctly created LanceRetriever")
    else:
        print(f"   ✗ Factory created {type(factory_retriever).__name__} instead")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Factory test failed: {e}")
    sys.exit(1)

# Clean up
print("\n6. Cleaning up test data...")
try:
    retriever.clear_collection()
    import shutil
    if Path("./lance_db_test").exists():
        shutil.rmtree("./lance_db_test")
    print("   ✓ Test data cleaned up")
except Exception as e:
    print(f"   ⚠ Cleanup warning: {e}")

print("\n" + "=" * 60)
print("✓ All verification tests passed successfully!")
print("=" * 60)
print("\nLanceDB is ready to use in your RAG system.")
print("\nNext steps:")
print("1. Update your .env file with VECTOR_STORE_TYPE=lancedb")
print("2. Run: python scripts/migrate_to_lancedb.py (if migrating from ChromaDB)")
print("3. Run: pytest tests/test_lancedb_retriever.py (for comprehensive tests)")
print("=" * 60)
