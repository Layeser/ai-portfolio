# Notes: one page per algorithm

For reading rather than running. Each page walks through the code in this repository (not the textbook version),
justifies its implementation choices with figures produced by `scripts/make_figures.py`, and states the limits of the
method.

| Algorithm | Page | Highlight |
|---|---|---|
| Linear regression | [linear_regression.md](linear_regression.md) | gradient descent and least squares agree; why the ridge option (coefficient spread 52 → 0.023) |
| Logistic regression | [logistic_regression.md](logistic_regression.md) | class weights on 5 % positives: recall 13 % → 89 % |
| k-NN | [knn.md](knn.md) | a prediction in three vectorised steps; choosing k |
| Gaussian naive Bayes | [naive_bayes.md](naive_bayes.md) | all classes in one expression; probabilities equal scikit-learn's to 1e-8 |
| Decision tree | [decision_tree.md](decision_tree.md) | split search in O(n log n); choosing the depth |
| PCA | [pca.md](pca.md) | two solvers, one answer; a deterministic sign |
| k-means | [kmeans.md](kmeans.md) | k-means++ and restarts: worse local minima 95 % → 4 % |
| Autoencoder | [autoencoder.md](autoencoder.md) | against PCA at equal code size: 57 % less reconstruction error at 16 numbers |
| Variational autoencoder | [vae.md](vae.md) | three changes that turn an autoencoder into a generator; the β trade-off measured |
| LeNet-5 | [lenet.md](lenet.md) | one image through the network, shape by shape; a schedule worth 0.4 point (99.20 %) |
| ResNet-18 | [resnet.md](resnet.md) | the degradation experiment reproduced: plain-34 trains worse than plain-18, ResNet-34 does not |
| EN→FR Transformer | [transformer.md](transformer.md) | translates sentences it never saw, and a measured case where it cannot |

Regenerate every figure and number:

```bash
PYTHONPATH=src python scripts/make_figures.py      # classical models: a few seconds on CPU
PYTHONPATH=src python scripts/make_dl_figures.py   # neural networks: MNIST / CIFAR-10, GPU recommended (see its header)
```

Values are also saved in `figures/<algo>/measurements.json`.
