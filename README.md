# CPS5801 Reproducibility Package

Contains frozen configurations, source adapter/evaluator/runner, tests, run-level results, audit notes, and figures. API-Bank data is not redistributed; see data/raw/README.md.

## Verified environment
Windows; Python 3.11.9; Ollama 0.40.16; NVIDIA RTX A5000 24 GB; CUDA 13.2. `pip check` clean; full test suite 27 passed in 2.57 s. Models: llama3.1:8b (digest prefix 46e0c10c039e) and qwen2.5:7b (845dbda0ea48).

## Reproduction
1. Install Python 3.11.9 and Ollama; create a virtual environment.
2. `python -m pip install -r metadata/requirements-lock.txt`
3. Set up the pinned API-Bank checkout as described in data/raw/README.md.
4. Pull the exact Ollama model tags and verify digests against the source metadata.
5. `python -m pytest -q`
6. `python scripts/validate_api_bank_sandbox.py`
7. Dry run: `python -m src.experiment_runner --formal-batch --dry-run`
8. Full rerun: `python -m src.experiment_runner --formal-batch` (126 runs; GPU time required).

Frozen design: 7 tasks × 2 models × 3 prompting packages × 3 repetitions; temperature 0, seed 42, sequential. Packages are bundled comparisons, not a factorial design. Never include credentials in logs.

Semantic review: all 126 formal records remain pending_human_review. Automated strict success is not semantic accuracy; do not report a semantic accuracy aggregate before documented human review and adjudication.
