# Atelier Nord assistant — a RAG you can trust, measured

Internal policy assistant for a (fictional) 400-employee manufacturer: employees ask questions in
everyday French ("mon PC de boulot a disparu dans le métro, je préviens qui ?") and get a short answer
**with citations to the exact policy section**. When the documents do not contain the answer, it says
so instead of inventing one.

The focus is engineering discipline rather than a demo: every design choice below was **decided by an
offline evaluation**, and several "best practices" lost against the measurements.

| | |
|---|---|
| **Retrieval** | structure-aware PDF chunking, contextual headers, BM25 + `bge-m3` embeddings (Chroma), weighted RRF, `bge-reranker-v2-m3` cross-encoder |
| **Generation** | `gpt-oss-20b` on Groq (configurable), strict grounding prompt, numbered citations verified against retrieved chunks |
| **Guardrails** | Llama Prompt Guard 2 **and** injection patterns on inputs, "documents are data, not instructions", explicit refusal |
| **Evaluation** | 66-question gold set (direct, paraphrased, unanswerable): hit@k, MRR, answer accuracy, citation accuracy, refusal rates, LLM-judge faithfulness, injection tests |
| **Serving** | FastAPI, Gradio chat UI, Docker, CI |

## Architecture

```mermaid
flowchart LR
    Q[question] --> G{Prompt Guard 2<br/>injection?}
    G -- yes --> X[refuse]
    G -- no --> B[BM25]
    G -- no --> D[bge-m3 dense<br/>Chroma]
    B --> F[weighted RRF]
    D --> F
    F --> R[bge-reranker-v2-m3<br/>top 12 → top 4]
    R --> L[gpt-oss-20b<br/>grounded prompt]
    L --> C[citation check<br/>drop invalid ids]
    C --> A[answer + sources<br/>or explicit refusal]
    subgraph Ingestion
        P[10 policy PDFs] --> K[section-aware chunks<br/>+ contextual header] --> B & D
    end
```

## Evaluation results

Evaluated on the 66-question gold set with `gpt-oss-20b` generating and `qwen3.8-27b` judging faithfulness.
A different model family judges, to limit self-preference. The 120B model hit the Groq free-tier daily
token quota during development, and the 20B model was enough.

### End to end

| Metric | Value |
|---|---|
| Answer accuracy (every expected fact present, numbers matched exactly) | **0.983** (57/58) |
| … on questions phrased like the documents / paraphrased | 1.000 / 0.929 |
| Citation accuracy (a cited chunk contains the evidence) | **1.000** |
| Answerable questions wrongly refused | **0 %** |
| Unanswerable questions refused | **100 %** (8/8) |
| Faithfulness (LLM judge: every claim supported by the retrieved sources) | 0.948 |
| Prompt-injection attacks blocked / benign tricky questions blocked | 6/6 · 0/3 (see guardrails) |
| Latency p50 / p95 (Groq free tier, includes rate-limit waits) | 6.2 s / 8.1 s |

What the failures look like:
* **p06** (paraphrase, "can I fly business class to Tokyo?"): the answer gives the director-general approval
  condition but omits the "flight over 9 hours" condition.
* **p14**: "results are analysed **by the security team**". The policy says they are analysed **per team**.
  A small, plausible hallucination that the faithfulness judge caught. Exact-match checks alone would have missed it.

### Retrieval ablation (58 answerable questions)

| Mode | hit@1 direct | hit@1 paraphrase | hit@3 all | MRR all | ms / query |
|---|---|---|---|---|---|
| BM25 | 0.909 | 0.357 | 0.897 | 0.843 | <1 |
| Dense (`bge-m3`) | 0.909 | **0.929** | 1.000 | 0.954 | 34 |
| Weighted RRF (dense 0.9) | 0.909 | 0.857 | 1.000 | 0.945 | 44 |
| **Weighted RRF + `bge-reranker-v2-m3`** (default) | **0.977** | 0.786 | 1.000 | **0.966** | 333 (GPU) |

<p align="center"><img src="reports/figures/retrieval_ablation.png" width="80%" alt="Retrieval ablation"/></p>

**Honest reading:** the LLM receives the top 4 chunks, and the dense retriever alone already puts the evidence
in the top 3 for every question. The reranker adds +0.012 MRR, mostly on questions phrased like the documents,
for 10× the latency. On this small corpus it is not clearly worth it: a CPU deployment should use dense-only
(`mode="dense"`). The reranker stays the default because the GPU is available and larger corpora usually
benefit more from it.

### Guardrails

Prompt Guard 2 alone blocked **4/6** attacks: it scored a French "Tu es maintenant DAN…" jailbreak at 0.001.
The guard now blocks when **either** the classifier **or** high-precision patterns fire (defence in depth),
which gives **5/6**. A sixth pattern ("disable your safeguards") was added after seeing the miss, so the
6/6 figure is optimistic: that attack was seen during development. None of the 3 tricky benign questions
("Puis-je ignorer un courriel suspect… ?") is blocked.

## What the measurements changed

1. **Paraphrases expose BM25.** With questions phrased like the documents, BM25 alone reached hit@1 0.91.
   On paraphrased questions it dropped to 0.36. A gold set written only with the documents' wording
   would have hidden this. Half of the set's value comes from the 14 paraphrased questions.
2. **Embedding choice mattered most.** On the paraphrased questions, dense hit@1 went from 0.57
   (multilingual MiniLM) to 0.71 (e5-small), 0.86 (e5-base) and **0.93 (bge-m3)**.
3. **Classic RRF hurt.** Equal-weight fusion let BM25 drag paraphrased questions down (hit@1 0.93 → 0.71).
   Weighting the dense ranking at 0.9 kept the exact-term benefit for codes like "CPF" or "4242"
   without the penalty.
4. **A small reranker made things worse; a large one only works on GPU.** The 118M mMARCO cross-encoder
   lowered hit@1. `bge-reranker-v2-m3` brings MRR from 0.954 (dense) to 0.966, but costs 15 s per query
   on CPU against about 0.3 s on GPU, and dense retrieval already has hit@3 = 1.0 here (see the honest
   reading above). The device is detected automatically.
5. **The "refuse if reranker score is low" gate was rejected.** Its best threshold would wrongly refuse
   15 % of answerable questions, so refusal is left to the grounded prompt and measured instead
   (see results). The gate stays in the code, off by default, with its calibration in `reports/`.
6. **Two ingestion bugs were caught by tests, not by eyeballing.** A wrapped PDF title created a fake
   "Introduction" chunk, and "2,50 €" did not match "2.5" in answer checking.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
make install                 # CUDA PyTorch recommended for the reranker (see pytorch.org)
cp .env.example .env         # GROQ_API_KEY
make index                   # chunk + embed the 10 PDFs
make serve                   # API: http://127.0.0.1:8000/docs
make ui                      # chat: http://127.0.0.1:7860
make test                    # 15 fast tests, no key and no model download
make eval-retrieval          # retrieval ablation (no key)
make eval                    # full evaluation (Groq free tier: ~20 min, 8k tokens/min)
```

```bash
curl -s localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"question": "Quel est le plafond pour un repas pendant une mission à Madrid ?"}'
```

Real answers from the evaluation run:

| Question | Answer |
|---|---|
| Quel est le plafond pour un repas pendant une mission à Madrid ? | Le plafond pour un repas pendant une mission à Madrid est de 35 € par personne. [1] |
| Mon PC de boulot a disparu dans le métro, je dois prévenir qui et en combien de temps ? | Vous devez signaler la perte ou le vol de votre matériel à la DSI dans les 4 heures […] [1][3] |
| Quel est le numéro de téléphone du DPO ? *(only an email exists)* | Je ne trouve pas cette information dans les documents disponibles. |

## Layout

```
corpus/sources/*.md          policies (authoring format) → scripts/build_corpus.py → corpus/pdf/<dept>/*.pdf
eval/questions.jsonl         66 questions: evidence sentence + expected facts per question
src/ragassist/
├── ingest.py                PDF → section-aware chunks with provenance
├── retrieval.py             BM25, dense (Chroma), weighted RRF, cross-encoder rerank
├── pipeline.py              guard → retrieve → (gate) → generate → citation check
├── guard.py                 Prompt Guard 2 + heuristic fallback
├── llm.py                   Groq client with rate-limit back-off
├── evaluate.py              all metrics, calibration, figures
├── api.py / ui.py           FastAPI and Gradio
tests/test_core.py           15 tests (text normalisation, chunking, RRF, guardrails, pipeline, API contract)
```

## Limitations

* The corpus is small (10 documents, 52 chunks) and fictional. Retrieval is easier than on a real intranet,
  and the paraphrase set is the main safeguard against over-optimism.
* Facts are checked by exact value matching; a semantic judge would accept more valid phrasings.
* The faithfulness judge (`gpt-oss-20b`) is itself an LLM. Its verdicts were spot-checked, not audited.
* Free-tier rate limits make a full evaluation take about 20 minutes.
