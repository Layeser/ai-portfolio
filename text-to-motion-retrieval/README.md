# Text-to-Motion Retrieval

Cross-modal retrieval: map **natural language** and **motion sequences** into a shared embedding space with a dual encoder.

## Results (validation)

| Metric | Value |
|--------|-------|
| Recall score (weighted) | **0.977** |
| Recall@1 | **0.933** |
| Recall@2 … @10 | 1.0 (best run) |

Architecture: **BERT** (`bert-base-uncased`) text encoder + **TransformerEncoder** motion encoder, cosine similarity ranking.

## Contents

| File | Role |
|------|------|
| `datachallenge.ipynb` | Main training / evaluation notebook |
| `text-motion-expl.ipynb` | Exploration / analysis |
| `submission_now_5.csv` / `ensemble_submission.csv` | Challenge submissions |
| `train.txt` | Train ID list (no raw motions) |

Raw motion tensors / checkpoints are **not** included (multi-GB). Request the challenge dataset separately or regenerate features from the original data challenge package.

## Tech stack

PyTorch · Hugging Face Transformers · NumPy · pandas · scikit-learn

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook datachallenge.ipynb
```

Place motion features under a local `data/` folder as expected by the notebook (see first cells).

## Improvements

- Extract a clean `train.py` from the notebook (production style)
- Add a small synthetic toy dataset for CI smoke tests
- Log metrics with MLflow

## Context

Data challenge / coursework project — kept here for the strong multimodal retrieval numbers and Transformer dual-encoder design.
