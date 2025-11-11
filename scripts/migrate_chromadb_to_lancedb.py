#!/usr/bin/env python3
"""
Migrate existing ChromaDB vector stores to LanceDB.

This utility script helps transition from ChromaDB to LanceDB by:
1. Checking for existing ChromaDB databases
2. Providing guidance on re-indexing with LanceDB
3. Cleaning up old ChromaDB files

Usage:
    python scripts/migrate_chromadb_to_lancedb.py --check
    python scripts/migrate_chromadb_to_lancedb.py --cleanup

Author: Declan McAlinden
Date: 2025-11-11
"""

import argparse
import logging
import shutil
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_chromadb_exists(project_root: Path) -> bool:
    """Check if ChromaDB directory exists."""
    chroma_dir = project_root / "chroma_db"
    if chroma_dir.exists():
        size_mb = sum(f.stat().st_size for f in chroma_dir.rglob('*') if f.is_file()) / (1024 * 1024)
        logger.info(f"✓ Found ChromaDB directory: {chroma_dir}")
        logger.info(f"  Size: {size_mb:.2f} MB")
        return True
    else:
        logger.info("✗ No ChromaDB directory found")
        return False


def check_lancedb_exists(project_root: Path) -> bool:
    """Check if LanceDB directory exists."""
    lance_dir = project_root / "lancedb"
    if lance_dir.exists():
        size_mb = sum(f.stat().st_size for f in lance_dir.rglob('*') if f.is_file()) / (1024 * 1024)
        logger.info(f"✓ Found LanceDB directory: {lance_dir}")
        logger.info(f"  Size: {size_mb:.2f} MB")
        return True
    else:
        logger.info("✗ No LanceDB directory found")
        return False


def cleanup_chromadb(project_root: Path, force: bool = False) -> None:
    """Remove ChromaDB directory."""
    chroma_dir = project_root / "chroma_db"
    
    if not chroma_dir.exists():
        logger.info("No ChromaDB directory to clean up")
        return
    
    if not force:
        response = input(f"\n⚠️  This will DELETE {chroma_dir} permanently. Continue? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Cleanup cancelled")
            return
    
    try:
        shutil.rmtree(chroma_dir)
        logger.info(f"✓ Deleted ChromaDB directory: {chroma_dir}")
    except Exception as e:
        logger.error(f"✗ Error deleting ChromaDB directory: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate from ChromaDB to LanceDB"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check for existing vector stores"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove ChromaDB directory (requires confirmation)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force cleanup without confirmation"
    )
    
    args = parser.parse_args()
    
    if args.check or (not args.cleanup):
        logger.info("="*80)
        logger.info("VECTOR STORE MIGRATION CHECK")
        logger.info("="*80)
        
        chromadb_exists = check_chromadb_exists(project_root)
        lancedb_exists = check_lancedb_exists(project_root)
        
        logger.info("\n" + "="*80)
        logger.info("MIGRATION STATUS")
        logger.info("="*80)
        
        if not chromadb_exists and not lancedb_exists:
            logger.info("✓ Clean state - no vector stores found")
            logger.info("  Run baseline evaluation to create LanceDB index")
        elif chromadb_exists and not lancedb_exists:
            logger.info("⚠️  ChromaDB found, LanceDB not yet created")
            logger.info("  ACTION REQUIRED:")
            logger.info("  1. Run: python scripts/run_baseline_evaluation.py --dataset squad --max_samples 100")
            logger.info("  2. This will create a new LanceDB index")
            logger.info("  3. Then run: python scripts/migrate_chromadb_to_lancedb.py --cleanup")
        elif chromadb_exists and lancedb_exists:
            logger.info("⚠️  Both vector stores exist")
            logger.info("  LanceDB is now active. You can safely remove ChromaDB:")
            logger.info("  Run: python scripts/migrate_chromadb_to_lancedb.py --cleanup")
        else:
            logger.info("✓ Migration complete - using LanceDB")
        
        logger.info("="*80 + "\n")
    
    if args.cleanup:
        logger.info("\n" + "="*80)
        logger.info("CLEANUP ChromaDB")
        logger.info("="*80)
        cleanup_chromadb(project_root, force=args.force)
        logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()
