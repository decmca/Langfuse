#!/usr/bin/env python3
"""
Fine-tune RAG model with PEFT/LoRA using batch processing.

This script demonstrates how to fine-tune the generator model on larger
datasets whilst maintaining efficient GPU utilisation.

Usage:
    python scripts/run_finetuning.py --dataset squad --max_samples 10000 --batch_size 16

Author: Declan McAlinden
Date: 2025-11-11
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset_loader import DatasetLoader
from src.tracking.langfuse_tracker import LangfuseTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RAGDataset(Dataset):
    """PyTorch Dataset for RAG fine-tuning."""
    
    def __init__(self, examples, tokenizer, max_length=1024):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Format as instruction-following task
        prompt = f"Question: {example.question}\n\nAnswer:"
        target = example.answer
        
        # Combine prompt and target
        full_text = f"{prompt} {target}"
        
        # Tokenise
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune RAG model with LoRA")
    
    parser.add_argument("--dataset", type=str, required=True, choices=["squad", "msmarco", "pubmedqa"])
    parser.add_argument("--max_samples", type=int, default=10000, help="Max training samples")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size per GPU")
    parser.add_argument("--gradient_accumulation", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--output_dir", type=str, default="./models/finetuned", help="Output directory")
    
    return parser.parse_args()


def main():
    """Main fine-tuning execution."""
    args = parse_args()
    
    logger.info(f"Fine-tuning on {args.dataset} with {args.max_samples} samples")
    logger.info(f"Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = DatasetLoader(str(project_root / "config" / "dataset_config.yaml"))
    train_data = dataset_loader.load_dataset(
        args.dataset,
        split="train",
        max_samples=args.max_samples
    )
    
    # Load model and tokeniser
    logger.info("Loading base model...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Configure LoRA
    logger.info("Configuring LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen architecture
        bias="none"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create dataset
    train_dataset = RAGDataset(train_data, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,  # Use bfloat16 for better stability
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=["tensorboard"],
        dataloader_num_workers=4,
        gradient_checkpointing=True  # Save memory
    )
    
    # Initialise trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("Fine-tuning complete!")


if __name__ == "__main__":
    main()
