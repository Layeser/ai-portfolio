# Sekou Kaba — AI Engineer portfolio

AI engineer (Centrale Lille × Centrale Casablanca double degree). I build AI systems end to end:
from data and models to APIs, monitoring and LLM agents, evaluated with numbers rather than demos.

[LinkedIn](https://www.linkedin.com/in/sekou-kaba-ai) · sekou.kaba.pro@gmail.com · Lille / Paris

---

## ⭐ Flagship projects

### [FraudGuard — fraud detection platform with an LLM investigation agent](./fraud-detection-platform/)

Point-in-time features → LightGBM → real-time FastAPI scoring with SHAP reasons → PSI drift monitoring →
**LangGraph + MCP agent** that investigates alerts, with deterministic guardrails on top of the LLM.

* PR-AUC **0.905** vs 0.644 (logistic) and 0.426 (expert rules) on a chronological test split
* `block` threshold: **92 % precision / 89 % recall**; drift monitor catches a new fraud pattern (PR-AUC 0.90 → 0.72) and triggers retraining
* Agent evaluated against ground truth: **85 % accuracy** on hard alerts. Its blind spot (stealth fraud) is documented, and a number-tracing guardrail catches LLM arithmetic errors
* `LightGBM` `FastAPI` `MLflow` `Prometheus` `LangGraph` `MCP` `Docker` `GitHub Actions` · 23 tests

### [Atelier Nord assistant — production-grade RAG with evaluation](./enterprise-rag-assistant/)

Grounded Q&A over internal HR / finance / IT / legal policies (French PDFs), with cited answers and
explicit refusal when the documents do not contain the answer.

* **98.3 %** answer accuracy and **100 %** correct citations on a 66-question gold set (incl. paraphrases); **0 %** false refusals and **8/8** unanswerable questions refused
* Every design choice decided by ablation: BM25 alone collapses on paraphrases (hit@1 0.36), `bge-m3` dense reaches 0.93, and the reranker's small gain is reported honestly
* Guardrails: Prompt Guard 2 + patterns (each missed attacks the other caught), LLM-judge faithfulness (caught a subtle hallucination)
* `bge-m3` `bge-reranker-v2-m3` `Chroma` `BM25` `Groq` `FastAPI` `Gradio` `Docker` · 15 tests

---

## Applied ML

| Project | Problem → solution | Key result |
|---|---|---|
| [ranking-bandits](./ranking-bandits/) | Which 5 items to show when click rates are unknown: cascading bandits learn online from clicks; benchmark + serving API | Thompson sampling: **99 %** of optimal CTR (3× random); documented negative result for "forgetting" |
| [hierarchical-product-nlp](./hierarchical-product-nlp/) | Product categorisation in a 184-node taxonomy: TF-IDF vs fine-tuned XLM-R, hierarchical metrics, human-in-the-loop routing, API | F1-macro 0.839 (notebook: 0.808); 82 % auto-categorised at 94.6 % accuracy |
| [text-to-motion-retrieval](./text-to-motion-retrieval/) | Retrieve motion-capture sequences from a description: CLIP-style dual encoder, search API; found and fixed a validation leak | **1st place** on the Kaggle leaderboard (weighted Recall@1–10: 0.898 vs 0.867); R@1 0.571 among 32 on **unseen** motions in the rebuilt pipeline |
| [visual-search](./visual-search/) | Image similarity search: 4 representations compared (pixels, autoencoder, self-supervised SimCLR, ImageNet transfer), FAISS index, search API | SimCLR without labels: P@10 **0.68** vs 0.35 for an autoencoder; ImageNet transfer 0.77 |
| [two-stage-detectors](./two-stage-detectors/) | R-CNN, Fast R-CNN and Faster R-CNN re-implemented on one ResNet-50 base and compared under the papers' protocol (VOC 2007) | Faster R-CNN **71.7 %** mAP (paper: 69.9 %), Fast 67.2 %, R-CNN 62.5 %; 12× faster from R-CNN to Faster |
| [sales-forecasting](./sales-forecasting/) | 28-day forecasts for 30,490 Walmart item×store series (M5) from sales, calendar and prices: LightGBM, TiDE (PyTorch), blend, forecast API | WRMSSE **0.555** on the M5 evaluation period (seasonal naive 0.847, winner 0.520) |
| [yolo-from-scratch](./yolo-from-scratch/) | YOLOv1 written from scratch in PyTorch; debugging three training bugs (neck, loss, optimizer) with an oracle check and overfit tests | VOC 2007 test mAP@0.5 **0.235** (from 0.0002 before the fixes), ResNet-50 ImageNet backbone, 25 epochs |

## Foundations

| Project | What it shows | Key result |
|---|---|---|
| [ml-from-scratch](./ml-from-scratch/) | Classical ML, VAE, ResNet and an EN→FR Transformer implemented from the maths, **verified against scikit-learn** | 13 tests, matches sklearn |

## LLM & agents

| Project | What it shows |
|---|---|
| [langchain-tool-agents](./langchain-tool-agents/) | Compact tool-calling agent (LangChain 1.x), offline timezone tool, tested end to end |
| [reddit-riddles-llm](./reddit-riddles-llm/) | LLM-based reference extraction and evaluation (semantic similarity, LLM judge) |

## Software

| Project | What it shows |
|---|---|
| [brainback](./brainback/) | Full-stack rewrite: Django REST Framework + Vue 3, Docker Compose, nginx |

---

## How the repository is organised

The applied projects (FraudGuard, RAG assistant, ranking-bandits, product categorisation, text-to-motion) share
the same layout, and each can be split into its own repository:

```
<project>/
├── README.md         problem, results with figures, design decisions, limitations
├── requirements.txt  dependencies (pip install -r requirements.txt into the shared venv)
├── Makefile          make test / train / evaluate / serve / download-weights (sets PYTHONPATH=src)
├── pytest.ini        test config (pythonpath = src, so no install step is needed)
├── src/<package>/    code (data → model → evaluation → API)
├── tests/            unit + contract tests, run in CI
└── reports/          figures and metrics produced by the code, versioned as evidence
```

* **CI**: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) tests every project in its own job.
* **Trained weights**: small models are versioned (FraudGuard); larger ones are attached to GitHub Releases with
  SHA-256 checksums (`make download-weights`), so every model can be used without retraining.
* **No client code or private data**: FraudGuard runs on a simulator, the RAG corpus is a fictional company,
  private training sets stay outside the repository.
