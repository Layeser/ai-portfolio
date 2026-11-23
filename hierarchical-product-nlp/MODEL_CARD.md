---
license: other  # trained on a company's private assignment data: keep this repository private
tags: [text-classification, hierarchical-classification, e-commerce, tf-idf, french]
---

# Product categorisation — TF-IDF + calibrated linear SVM

Leaf-category classifier (100 leaves of a 184-node taxonomy) for French / German / English product records.
Code: https://github.com/Layeser/ai-portfolio/tree/main/hierarchical-product-nlp

Test set: accuracy 0.869, macro-F1 0.839, hierarchical F1 0.932; 82 % of products auto-categorised at 94.6 %
accuracy using the threshold in `tfidf/threshold.json` (calibrated on validation for 95 %).

**Private on purpose:** the TF-IDF vocabulary stores words and brands from the private training data.

```bash
python scripts/weights.py download   # needs access to this private repository (HF_TOKEN)
uvicorn prodcat.api:app
```
