---
license: other  # trained on data-challenge data: check its terms before reuse
tags: [text-to-motion, retrieval, contrastive-learning, motion-capture, pytorch]
---

# Text-to-motion dual encoder

CLIP-style dual encoder that maps English descriptions of two-person interactions and motion-capture
sequences (T × 384 features) into a shared 256-d space. Code, training and evaluation:
https://github.com/Layeser/ai-portfolio/tree/main/text-to-motion-retrieval

| Held-out protocol (601 unseen motions) | R@1 | R@5 | R@10 |
|---|---|---|---|
| target among 32 candidates | 0.571 | 0.885 | 0.952 |
| full gallery of 601 motions | 0.119 | 0.367 | 0.500 |

Files: `dual_encoder.pt` (state dict: all-MiniLM-L6-v2 text tower fine-tuned at lr×0.1, 4-layer motion
Transformer, projections, temperature), `normalizer.npz` (per-feature mean / std from the training split; apply
it to motions before encoding), `split.json` (train / held-out motion ids), `metrics.json`.

```python
python scripts/weights.py download   # in the repository: fetches and verifies these files
uvicorn t2m.api:app                  # search service, no training
```

Limitations: trained on the data-challenge motion set (6,018 sequences); the motion feature layout is the
challenge's own 384-d format, so other mocap formats need conversion.
