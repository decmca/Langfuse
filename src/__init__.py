"""
Citation-Aware Q&A System with Langfuse Tracking.

A comprehensive RAG evaluation framework for demonstrating iterative
improvements through fine-tuning, prompt engineering, and retrieval optimisation.
"""

__version__ = "1.0.0"
__author__ = "Declan McAlinden"

# Workaround for ChromaDB SQLite version requirement on older systems
# Replaces system sqlite3 with pysqlite3-binary (includes SQLite >=3.35.0)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
