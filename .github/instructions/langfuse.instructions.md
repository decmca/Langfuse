Always use British UK grammar.
This is the Langfuse open source code: https://github.com/langfuse/langfuse
This is the repository for my project: https://github.com/decmca/Langfuse

## Project Summary

The [decmca/Langfuse](https://github.com/decmca/Langfuse) repository is a portfolio project demonstrating systematic improvement of Retrieval-Augmented Generation (RAG) systems through fine-tuning, prompt engineering, and comprehensive observability tracking using the Langfuse platform.

### Core Purpose

This is a citation-aware question-answering system that evaluates and improves RAG performance across multiple datasets (SQuAD, Natural Questions) whilst maintaining detailed observability through [Langfuse integration](https://github.com/decmca/Langfuse/blob/main/requirements.txt). The project serves as a demonstration of methodical LLM improvement techniques.

### Project Structure

The repository follows a modular architecture with distinct components:

- **[Data handling](https://github.com/decmca/Langfuse/tree/main/src/data)**: Dataset downloading and preprocessing capabilities
- **[Models](https://github.com/decmca/Langfuse/tree/main/src/models)**: Base RAG implementations and fine-tuned variants
- **[Evaluation](https://github.com/decmca/Langfuse/tree/main/src/evaluation)**: Metrics computation (ROUGE, BERTScore, citation accuracy)
- **[Tracking](https://github.com/decmca/Langfuse/tree/main/src/tracking)**: Langfuse observability integration
- **[Prompts](https://github.com/decmca/Langfuse/tree/main/src/prompts)**: Prompt engineering templates and experiments
- **[Scripts](https://github.com/decmca/Langfuse/tree/main/scripts)**: Automation for dataset preparation and baseline evaluation

### Key Technologies

The project utilises PyTorch, Transformers, ChromaDB for vector storage, FAISS for similarity search, and Sentence Transformers for embeddings. Fine-tuning is implemented using [PEFT (Parameter-Efficient Fine-Tuning) with LoRA](https://github.com/decmca/Langfuse/blob/main/requirements.txt), whilst DeepSpeed enables efficient training at scale.

### Quick Start Workflow

Setup involves creating a virtual environment, installing dependencies from [requirements.txt](https://github.com/decmca/Langfuse/blob/main/requirements.txt), configuring API keys in `.env`, downloading datasets (SQuAD by default), and running baseline evaluations with configurable sample sizes.

### Configuration Requirements

The project requires Langfuse API credentials (public and secret keys), HuggingFace tokens for gated models, and GPU configuration settings. Directory paths for ChromaDB storage and model caching must be specified in the [environment configuration](https://github.com/decmca/Langfuse/blob/main/.env.example).