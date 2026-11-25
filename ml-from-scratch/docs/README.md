# Notes: one page per algorithm

For reading rather than running. Each page explains the code in this repository (not the textbook version), shows
where it works and where it breaks with figures produced by `scripts/make_figures.py`, and lists what this
implementation does not handle compared with a production library.

| Algorithm | Page | Where it breaks |
|---|---|---|
| PCA | [pca.md](pca.md) | unscaled features (Wine: one feature takes 99.8 % of the variance); non-linear structure |
| k-means | [kmeans.md](kmeans.md) | non-convex clusters (two moons); one random initialisation (8 blobs) |

Regenerate every figure and number:

```bash
PYTHONPATH=src python scripts/make_figures.py   # a few seconds on CPU; values also in figures/<algo>/measurements.json
```
