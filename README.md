# Citation-Aware Q&A System with Langfuse Tracking

> Portfolio Project: Demonstrating systematic RAG improvement through fine-tuning, prompt engineering, and observability

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/download_datasets.py --datasets squad
python scripts/run_baseline_evaluation.py --dataset squad --max_samples 1000
```

See detailed documentation in README-complete.md (optional).
