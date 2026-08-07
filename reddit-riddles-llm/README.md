# Reddit Riddles — LLM Answer Extraction

Pipeline to build **riddle → answer** pairs from the Hugging Face dataset `sileod/reddit-riddles`, then evaluate extraction strategies (heuristic scoring vs LLM-based selection).

## Highlights

- Preprocessing + multiple extraction strategies
- Parallel LLM extraction (`parallel_extraction.py`) via OpenRouter (`API_KEY` env)
- Semantic evaluation with MiniLM embeddings
- Sample evaluation JSON artifacts included

## Results (sample evaluation)

LLM-based selection improved high-similarity matches vs score-based heuristics on evaluated samples (see notebooks / `evaluation_results_*.json`).

## Files

| File | Role |
|------|------|
| `reddit_riddle.ipynb` | Main exploration / extraction |
| `reddit_riddles_evaluation.ipynb` | Evaluation notebook |
| `parallel_extraction.py` | Concurrent LLM extraction |
| `updated_extract_function.py` | Extraction helpers |
| `pairs_llm_based_*.json` | Example extracted pairs |
| `evaluation_results_*.json` | Example metrics dumps |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # API_KEY=...
jupyter notebook reddit_riddle.ipynb
```

## Security

- Keys **must** come from `.env` (`os.getenv`)
- Do **not** publish older notebooks that hardcode OpenRouter keys; rotate any leaked key
