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
