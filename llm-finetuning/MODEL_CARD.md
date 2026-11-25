# LoRA adapter — Qwen2.5-1.5B-Instruct for receipt → JSON extraction

LoRA adapter (rank 16, alpha 32, every attention and MLP projection; 18.5 M trainable parameters, 1.2 % of the
model) for [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), trained to turn the text
of a receipt into structured JSON (items, quantities, prices, subtotal, tax, total, payment).

| Test split of CORD v2 (100 receipts) | base model, zero-shot | + this adapter |
|---|---|---|
| Field F1 | 0.543 | **0.870** |
| Receipts with every field right | 0 % | **45 %** |
| Generation time per receipt (RTX 5060 laptop, batch 16) | 1.97 s | **0.45 s** |

**Use it** (from `llm-finetuning/`):

```bash
make download-weights        # adapter -> artifacts/lora, SHA-256 verified
make serve                   # POST /extract {"text": "..."}; the adapter is merged into the base model at start-up
```

The base model is downloaded from the Hugging Face Hub on first use. Training and evaluation code, protocol and
limits: [llm-finetuning/README.md](https://github.com/Layeser/ai-portfolio/tree/main/llm-finetuning).

**Data and licences.** Trained on the annotations of CORD v2 (Park et al., 2019), CC-BY-4.0. The base model is
released under Apache 2.0. The input is text rebuilt from annotated words, not OCR output: expect lower accuracy on raw
OCR text.
