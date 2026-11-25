# k-nearest neighbours

> Classifies a point by the majority label of its *k* closest training points: no training, a flexible boundary, and a
> strong baseline on small, low-dimensional data.

k-NN has no parameters to learn: `fit` stores the training set and encodes the labels. All the work happens at
prediction time, in three steps.

## How a prediction is made

From [`src/mlscratch/classical/knn.py`](../src/mlscratch/classical/knn.py).

**1. Distances to every stored point**, by expanding the square into matrix products. Queries are processed in
batches of `batch_size`, so memory never holds a (queries × points × features) array.

```math
\lVert q - x_i \rVert^2 = \lVert q \rVert^2 - 2\, q^\top x_i + \lVert x_i \rVert^2
```

```python
d2 = (A**2).sum(axis=1)[:, None] - 2 * A @ B.T + (B**2).sum(axis=1)[None, :]
return np.maximum(d2, 0.0)
```

**2. The k nearest, without a full sort.** `argpartition` isolates the *k* smallest distances in linear time per row;
their order does not matter for a vote.

```python
nn = np.argpartition(d2, self.k - 1, axis=1)[:, : self.k]  # k smallest, unordered: O(n) per row
```

**3. A vote**, uniform or weighted by the inverse distance ($w_i = 1 / \lVert q - x_i \rVert$, an exact match winning
outright). `np.add.at` accumulates each neighbour's weight into its class:

```python
votes = np.zeros((len(Q), len(self.classes_)))
np.add.at(votes, (np.arange(len(Q))[:, None], labels), w)
return votes.argmax(axis=1)  # argmax returns the first maximum: smallest label wins ties
```

Ties between classes go to the smallest label, as in scikit-learn.

## Choosing k

<p align="center"><img src="../figures/knn/k_smoothness.png" width="95%" alt="k = 1 and k = 15 on noisy moons"></p>

Noisy moons, 1 000 points. With $k = 1$ the boundary wraps every training point (train 1.00, test 0.86); with $k = 15$
it smooths out and does better on unseen data (train 0.93, test 0.90). *k* is the bias–variance dial of the method.

## Where k-NN fits, and where it does not

It needs features on comparable scales, since distances mix units, and it degrades when many dimensions are
irrelevant, because noise enters every distance with the same weight as signal. The model is the whole training set,
so memory and prediction time grow with it: every query scans all stored points. Production systems answer that
with indexes, KD-trees or ball trees in low dimensions, approximate nearest-neighbour search (HNSW, quantised inverted
files as in FAISS) in high dimensions.

## Tests

From [`tests/test_against_sklearn.py`](../tests/test_against_sklearn.py), on Iris:

```python
ours = KNearestNeighbors(k=5).fit(X_tr, y_tr).predict(X_te)
ref = KNeighborsClassifier(n_neighbors=5, algorithm="brute").fit(X_tr, y_tr).predict(X_te)
assert (ours == ref).mean() >= 0.97  # only distance ties may break differently
```

[`tests/test_extras.py`](../tests/test_extras.py): distance-weighted votes, computed in batches of eight queries, agree
with scikit-learn's on at least 98 % of points.

<sub>Figure and numbers: `PYTHONPATH=src python scripts/make_figures.py`
([`figures/knn/measurements.json`](../figures/knn/measurements.json)).</sub>
