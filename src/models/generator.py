# (See earlier content) — Full file inserted
"""
LLM generator component for citation-aware answer generation.

Supports HuggingFace transformers with citation extraction and formatting.

Author: Declan McAlinden
Date: 2025-11-10
"""

import logging
import re
import time
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class Generator:
    """
    LLM-based answer generator with citation awareness.
    
    This class handles:
    - Loading and managing LLM models (Qwen, DeepSeek, Llama)
    - Generating answers from retrieved contexts
    - Formatting responses with proper citations
    - Tracking token usage and generation metrics
    
    Attributes:
        model_name (str): Name/path of the LLM model
        model: Loaded transformer model
        tokenizer: Tokenizer for the model
        device (str): Compute device ("cuda" or "cpu")
        max_new_tokens (int): Maximum tokens to generate
    """
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 512
    ):
        """
        Initialise LLM generator with mixed precision and quantisation support.
        
        Args:
            model_name: HuggingFace model name or local path
            device: Compute device ("cuda", "cpu", or None for auto-detect)
            load_in_8bit: Whether to use 8-bit quantisation (reduces memory by ~50%, 
                         achieves 3-5x speedup with minimal accuracy loss)
            load_in_4bit: Whether to use 4-bit quantisation (even more aggressive 
                         memory reduction, requires bitsandbytes)
            torch_dtype: Data type for model weights:
                        - "bfloat16" (recommended): Best for A100/H100/Quadro RTX, 
                          excellent numerical stability, ~50% memory reduction
                        - "float16": Compatible with most GPUs, but can have 
                          numerical overflow on some models
                        - "float32": Full precision, slower but most stable
            max_new_tokens: Maximum number of tokens to generate
        
        Notes:
            - BF16 is preferred for Quadro RTX 6000 and newer GPUs
            - Use batch sizes that are multiples of 8 for optimal tensor core utilisation
            - Quantisation (8-bit/4-bit) requires the bitsandbytes library
        
        Raises:
            RuntimeError: If model loading fails
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Loading generator model: {model_name}")
        logger.info(f"Using device: {self.device}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Set padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Convert torch_dtype string to actual dtype
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16
            }
            torch_dtype_actual = dtype_map.get(torch_dtype, torch.bfloat16)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype_actual,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            # Move to device if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load generator model '{model_name}': {str(e)}")
    
    def generate(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate answer with citations based on query and retrieved contexts.
        
        Args:
            query: User question
            contexts: List of retrieved context dicts with "text" and "score" keys
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling (vs. greedy decoding)
            system_prompt: Optional system prompt for instruction-tuned models
        
        Returns:
            Tuple of (generated_answer, metrics_dict)
            - generated_answer: Answer text with citations
            - metrics_dict: Token counts, latency, GPU memory usage
        """
        start_time = time.time()
        
        # Format prompt with contexts
        formatted_prompt = self._format_prompt(query, contexts, system_prompt)
        
        # Tokenise input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.model.config.max_position_embeddings - self.max_new_tokens
        ).to(self.device)
        
        input_token_count = inputs["input_ids"].shape[1]
        
        # Track GPU memory before generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_before = torch.cuda.max_memory_allocated() / 1e9
        else:
            gpu_memory_before = 0
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Track GPU memory after generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_after = torch.cuda.max_memory_allocated() / 1e9
            gpu_memory_used = gpu_memory_after - gpu_memory_before
        else:
            gpu_memory_used = 0
        
        # Decode output
        generated_text = self.tokenizer.decode(
            outputs[0][input_token_count:],  # Skip input tokens
            skip_special_tokens=True
        )
        
        output_token_count = outputs.shape[1] - input_token_count
        
        # Extract citations from generated text
        citations = self._extract_citations(generated_text)
        
        # Calculate metrics
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        metrics = {
            "token_counts": {
                "input": input_token_count,
                "output": output_token_count
            },
            "latency_ms": latency_ms,
            "gpu_memory_used_gb": gpu_memory_used,
            "citations_count": len(citations)
        }
        
        logger.debug(f"Generated {output_token_count} tokens in {latency_ms:.2f}ms")
        
        return generated_text.strip(), metrics
    
    def generate_batch(
        self,
        queries: List[str],
        contexts_list: List[List[Dict[str, Any]]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: Optional[str] = None
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Generate answers for multiple queries in batch for improved GPU utilisation.
        
        Args:
            queries: List of user questions
            contexts_list: List of retrieved context lists (one per query)
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling threshold
            do_sample: Whether to use sampling (vs. greedy decoding)
            system_prompt: Optional system prompt for instruction-tuned models
        
        Returns:
            List of tuples (generated_answer, metrics_dict) for each query
        """
        if len(queries) != len(contexts_list):
            raise ValueError("Number of queries must match number of context lists")
        
        start_time = time.time()
        batch_size = len(queries)
        
        # Format prompts for all queries
        formatted_prompts = [
            self._format_prompt(q, ctx, system_prompt)
            for q, ctx in zip(queries, contexts_list)
        ]
        
        # Configure tokeniser for batch processing
        # Left padding is crucial for causal LM batch generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        
        # Tokenise with padding to longest sequence in batch
        inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding="longest",  # Pad all sequences to match longest
            truncation=True,
            max_length=self.model.config.max_position_embeddings - self.max_new_tokens,
            return_attention_mask=True
        ).to(self.device)
        
        input_token_counts = inputs["attention_mask"].sum(dim=1).cpu().tolist()
        
        # Track GPU memory before generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_before = torch.cuda.max_memory_allocated() / 1e9
        else:
            gpu_memory_before = 0
        
        # Batch generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs["attention_mask"]
            )
        
        # Track GPU memory after generation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory_after = torch.cuda.max_memory_allocated() / 1e9
            gpu_memory_used = (gpu_memory_after - gpu_memory_before) / batch_size
        else:
            gpu_memory_used = 0
        
        # Decode outputs for each item in batch
        results = []
        for i in range(batch_size):
            input_len = input_token_counts[i]
            
            # Extract generated tokens (skip input)
            generated_tokens = outputs[i][input_len:]
            
            generated_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            output_token_count = len(generated_tokens)
            
            # Extract citations
            citations = self._extract_citations(generated_text)
            
            # Per-example metrics
            metrics = {
                "token_counts": {
                    "input": input_len,
                    "output": output_token_count
                },
                "citations_count": len(citations),
                "gpu_memory_used_gb": gpu_memory_used  # Averaged across batch
            }
            
            results.append((generated_text.strip(), metrics))
        
        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side
        
        # Calculate batch-level timing
        end_time = time.time()
        batch_latency_ms = (end_time - start_time) * 1000
        avg_latency_ms = batch_latency_ms / batch_size
        
        # Update metrics with timing information
        for i in range(batch_size):
            results[i][1]["latency_ms"] = avg_latency_ms
            results[i][1]["batch_latency_ms"] = batch_latency_ms
            results[i][1]["batch_size"] = batch_size
        
        logger.debug(
            f"Generated {batch_size} answers in {batch_latency_ms:.2f}ms "
            f"({avg_latency_ms:.2f}ms per example)"
        )
        
        return results
    
    def _format_prompt(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Format prompt with query and contexts.
        
        Args:
            query: User question
            contexts: Retrieved contexts with text and scores
            system_prompt: Optional system instruction
        
        Returns:
            Formatted prompt string
        """
        # Default system prompt for citation-aware generation
        if system_prompt is None:
            system_prompt = (
                "You are a helpful AI assistant that answers questions based on provided contexts. "
                "Always cite your sources using [1], [2], etc. to reference the context passages. "
                "Only include information that is supported by the provided contexts."
            )
        
        # Format contexts with citation numbers
        context_str = "\n\n".join([
            f"[{i+1}] {ctx['text']}"
            for i, ctx in enumerate(contexts)
        ])
        
        # Build prompt (adjust format based on model type)
        if "qwen" in self.model_name.lower():
            # Qwen format
            prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            prompt += f"<|im_start|>user\n"
            prompt += f"Contexts:\n{context_str}\n\n"
            prompt += f"Question: {query}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n"
        
        elif "llama" in self.model_name.lower():
            # Llama format
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>"
            prompt += f"<|start_header_id|>user<|end_header_id|>\n"
            prompt += f"Contexts:\n{context_str}\n\n"
            prompt += f"Question: {query}<|eot_id|>"
            prompt += f"<|start_header_id|>assistant<|end_header_id|>\n"
        
        else:
            # Generic format
            prompt = f"System: {system_prompt}\n\n"
            prompt += f"Contexts:\n{context_str}\n\n"
            prompt += f"Question: {query}\n\n"
            prompt += f"Answer:"
        
        return prompt
    
    def _extract_citations(self, text: str) -> List[int]:
        """
        Extract citation numbers from generated text.
        
        Args:
            text: Generated answer text
        
        Returns:
            List of citation numbers found in text
        """
        # Find all [1], [2], etc. patterns
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, text)
        
        # Convert to integers and remove duplicates
        citations = sorted(set(int(m) for m in matches))
        
        return citations
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "max_new_tokens": self.max_new_tokens,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "dtype": str(self.model.dtype)
        }
