# Reddit Riddles — LLM Answer Extraction

Pipeline to build **riddle → answer** pairs from the Hugging Face dataset `sileod/reddit-riddles`, then evaluate extraction strategies (heuristic scoring vs LLM-based selection).

## Highlights

- Preprocessing + multiple extraction strategies
- Parallel LLM extraction (`src/parallel_extraction.py`) via OpenRouter (`API_KEY` env)
- Semantic evaluation with MiniLM embeddings
- Sample evaluation JSON artifacts included

## Results

`gpt-4o-mini` answers the riddles; its answers are scored against the reference answer extracted from the
Reddit thread. The same model answers in both runs, so the only change is **how the reference is extracted**
(mapping from the original file timestamps: the heuristic-reference run, saved at 16:53, predates the LLM-extracted
pairs written at 16:55; the LLM-reference run was saved at 16:59).

| Reference extraction | Riddles | Semantic similarity | Token F1 | Keyword overlap | LLM-judge score |
|---|---|---|---|---|---|
| Score-based heuristic (top-voted comment) | 1,000 | 0.349 | 0.155 | 0.181 | 0.296 |
| **LLM-based selection** | 823 | **0.402** | **0.190** | **0.214** | **0.389** |

Exact match is 0 in both runs: riddle answers are phrased freely ("an eye" vs "the human eye"), which is why
semantic and judge-based metrics are needed. The ~30 % relative gain in judge score most likely comes from **cleaner
references**: the heuristic often picked a long, chatty comment rather than the answer itself. Caveat: the
LLM run covers 823 riddles instead of 1,000, so the two samples are not identical.

## Files

| Path | Role |
|------|------|
| `notebooks/reddit_riddle.ipynb` | Exploration and reference extraction |
| `notebooks/reddit_riddles_evaluation.ipynb` | Evaluation (semantic similarity, token F1, LLM judge) |
| `src/parallel_extraction.py` | Concurrent LLM extraction |
| `results/pairs_llm_based.json` | Riddle → answer pairs extracted by the LLM |
| `results/evaluation_heuristic_references.json` | Scores against heuristic references (1,000 riddles) |
| `results/evaluation_llm_references.json` | Scores against LLM-extracted references (823 riddles) |
| `docs/dataset.md` | Dataset structure: posts, comments, how answers are validated |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # API_KEY=...
jupyter notebook notebooks/reddit_riddle.ipynb
```

## Security

- Keys **must** come from `.env` (`os.getenv`)
- Do **not** publish older notebooks that hardcode OpenRouter keys; rotate any leaked key
