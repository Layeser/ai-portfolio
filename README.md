# Portfolio — Sekou Kaba (public-ready)

Only this folder is safe to publish. Prefer **one GitHub repo per subfolder**.

## Machine Learning / Research

| Folder | Focus |
|--------|--------|
| [`spiking-transformers/`](./spiking-transformers/) | HP-STAtten Spiking Transformers (91% CIFAR-10) |
| [`yolo-from-scratch/`](./yolo-from-scratch/) | YOLOv1 end-to-end in PyTorch |
| [`text-to-motion-retrieval/`](./text-to-motion-retrieval/) | BERT + motion Transformer (R@1 0.933) |
| [`rl-bandits-fqi/`](./rl-bandits-fqi/) | Bandits + Fitted Q-Iteration (course) |
| [`cv-deep-labs/`](./cv-deep-labs/) | CV labs: keypoints, flow nets, unsupervised features |
| [`ml-from-scratch/`](./ml-from-scratch/) | Classical ML + AE/VAE + LeNet/ResNet + EN→FR Transformer |

## NLP / LLM / Agents

| Folder | Focus |
|--------|--------|
| [`rag-pdf-fastapi/`](./rag-pdf-fastapi/) | FastAPI RAG (FAISS + MiniLM + LLM) |
| [`langchain-tool-agents/`](./langchain-tool-agents/) | Tool-calling & ReAct agents |
| [`reddit-riddles-llm/`](./reddit-riddles-llm/) | LLM answer extraction + eval |
| [`hierarchical-product-nlp/`](./hierarchical-product-nlp/) | Product taxonomy (F1-macro 0.81) |

## MLOps / Software

| Folder | Focus |
|--------|--------|
| [`mlops-fastapi-ci/`](./mlops-fastapi-ci/) | Train → MLflow → FastAPI → Docker → CI |
| [`brainback/`](./brainback/) | Django REST + Vue 3 fullstack |

## Never publish

- `../vigicrg/` (banking IP)
- `../masterds/` raw dump (datasets, checkpoints, secrets)
- Any `.env` or `sk-or-…` keys

## Before each `git push`

From a project folder, confirm there is no `.env`, no `*.pt` / `*.pkl`, and no hardcoded `sk-or-v1-…` keys.
