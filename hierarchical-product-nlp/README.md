# Hierarchical Product Categorization (NLP)

Technical assignment: predict **leaf product categories** from e-commerce text using classical ML then a Transformer + CNN head.

## Results

| Model | Test F1-macro | Test Acc. (micro) |
|-------|---------------|-------------------|
| TF-IDF + Logistic Regression | **0.800** | 0.835 |
| MiniLM + multi-n-gram CNN | **0.808** | 0.842 |

## Contents

| File | Role |
|------|------|
| `project.ipynb` | Full solution notebook |
| `tree.json` | Category taxonomy |
| `arch.png` / `archi.png` | Architecture diagrams |
| `Guidelines.pdf` | Assignment brief |
| `requirements.txt` | Dependencies |

## Stack

scikit-learn · PyTorch · Hugging Face Transformers (`all-MiniLM-L6-v2`) · spaCy · pandas

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook project.ipynb
```

Training CSVs / `.pth` checkpoints are **not** included (large). Place data next to the notebook as described in the first cells, or adapt paths.

## Why publish

Clean applied NLP signal: hierarchical labels, strong baseline, light fine-tuning, measurable F1 — good complement to RAG / agents repos.
