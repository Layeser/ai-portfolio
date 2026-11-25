# Notes: one page per algorithm

For reading rather than running. Each page explains the code in this repository (not the textbook version), shows
where it works and where it breaks with figures produced by `scripts/make_figures.py`, and lists what this
implementation does not handle compared with a production library.

| Algorithm | Page | Where it breaks |
|---|---|---|
| Linear regression | [linear_regression.md](linear_regression.md) | collinear features (std of a coefficient: 52 without ridge); unscaled features and gradient descent |
| Logistic regression | [logistic_regression.md](logistic_regression.md) | separable data (weights grow forever); 5 % positives (13 % recall) |
| k-NN | [knn.md](knn.md) | query time grows with n; irrelevant features (0.97 → 0.81) |
| Gaussian naive Bayes | [naive_bayes.md](naive_bayes.md) | correlated features (log loss 0.80 → 2.77); a class made of two blobs |
| Decision tree | [decision_tree.md](decision_tree.md) | depth (test 0.91 → 0.81); diagonal boundaries |
| PCA | [pca.md](pca.md) | unscaled features (Wine: one feature takes 99.8 % of the variance); non-linear structure |
| k-means | [kmeans.md](kmeans.md) | non-convex clusters (two moons); local optima (95 % of single random starts) |

Regenerate every figure and number:

```bash
PYTHONPATH=src python scripts/make_figures.py   # a few seconds on CPU; values also in figures/<algo>/measurements.json
```
