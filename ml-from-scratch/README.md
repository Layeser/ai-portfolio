# ML from Scratch

Implementations you can **read in one sitting**. No black-box `sklearn.fit`.  
NumPy for classical ML. PyTorch only where tensors / autograd earn their keep.

Inspired by the teaching style of [Andrej Karpathy](https://karpathy.ai/) — small models, clear math, runnable demos.

Each algorithm gets a one-page note in [`docs/`](docs/README.md): the maths as implemented, real code snippets, measured
figures, and the limits of the method.

## Philosophy

1. **Own the math** — every update rule is written out (gradients, EM-style steps, ELBO, attention masks).
2. **Classic datasets** — Iris / make\_\* / MNIST / CIFAR-10. Sklearn is used for *data*, never for the model.
3. **One file ≈ one idea** — open a script, run it, see a plot or a metric.
4. **Ship something that translates** — a tiny EN→FR Transformer on a **handcrafted** parallel corpus, decoded autoregressively.

## Layout

```text
ml-from-scratch/src/mlscratch/
├── classical/          # NumPy
│   ├── linear_regression.py
│   ├── logistic_regression.py
│   ├── knn.py
│   ├── naive_bayes.py
│   ├── decision_tree.py
│   ├── kmeans.py
│   └── pca.py
├── deep_learning/      # PyTorch
│   ├── autoencoder.py
│   ├── vae.py
│   ├── lenet.py
│   └── resnet.py
├── nlp/
│   └── en_fr_translation/   # toy parallel data + seq2seq Transformer
└── utils/common.py
```

## Setup

```bash
cd ml-from-scratch
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && export PYTHONPATH=src
```

## Teaching figures

```bash
PYTHONPATH=src python scripts/make_figures.py      # classical models: deterministic, a few seconds on CPU
PYTHONPATH=src python scripts/make_dl_figures.py   # neural networks: ~1 h on a laptop GPU (ResNet runs cached)
```

## Correctness: checked against scikit-learn

Writing an algorithm from the maths only counts if it gives the right answer. `tests/` (42 tests, ~15 s on CPU,
run in CI) compares every classical model with its scikit-learn reference, with explicit tolerances:

| Model | Check |
|---|---|
| Linear regression | closed form = sklearn coefficients (rtol 1e-6); gradient descent converges to the closed form |
| PCA | explained variance identical; components identical up to sign |
| k-NN, Gaussian naive Bayes | ≥ 97 % identical predictions (only distance ties may differ) |
| Logistic regression, decision tree | test accuracy within 2 / 5 points of sklearn on breast cancer |
| k-means | ARI > 0.95 on separated blobs; inertia within 5 % of sklearn's |
| LeNet, ResNet-18, autoencoder, VAE | output shapes; KL term = 0 for a standard-normal posterior; each model overfits a tiny batch (classic bug detector) |
| Deep-learning additions | augmentation is a true crop/flip, β scales only the KL term, attention rows sum to 1 and ignore padding, held-out sentences never leak into training |

```bash
pip install -r requirements.txt
python -m pytest -q                                          # all tests
python -m pytest -q tests/test_against_sklearn.py            # only the comparisons with scikit-learn
python -m pytest -v tests/test_against_sklearn.py -k pca     # one model, with details
```

## Classical ML (NumPy)

| Algorithm | Dataset | Run | Notes |
|-----------|---------|-----|-------|
| Linear regression | `make_regression` | `python -m mlscratch.classical.linear_regression` | [page](docs/linear_regression.md) |
| Logistic regression | `make_classification` | `python -m mlscratch.classical.logistic_regression` | [page](docs/logistic_regression.md) |
| k-NN | Iris | `python -m mlscratch.classical.knn` | [page](docs/knn.md) |
| Gaussian Naive Bayes | Iris | `python -m mlscratch.classical.naive_bayes` | [page](docs/naive_bayes.md) |
| Decision tree (CART, Gini) | Iris | `python -m mlscratch.classical.decision_tree` | [page](docs/decision_tree.md) |
| k-Means | blobs | `python -m mlscratch.classical.kmeans` | [page](docs/kmeans.md) |
| PCA | Iris (2D viz) | `python -m mlscratch.classical.pca` | [page](docs/pca.md) |

Plots land in `outputs/<algo>/`.

## Deep Learning (PyTorch)

| Model | Task | Run | Notes |
|-------|------|-----|-------|
| Autoencoder | MNIST reconstruction | `python -m mlscratch.deep_learning.autoencoder` | [page](docs/autoencoder.md) |
| VAE | MNIST generation | `python -m mlscratch.deep_learning.vae` | [page](docs/vae.md) |
| LeNet-5 | MNIST classification | `python -m mlscratch.deep_learning.lenet` | [page](docs/lenet.md) |
| ResNet-18 (CIFAR variant) | CIFAR-10 | `python -m mlscratch.deep_learning.resnet` | [page](docs/resnet.md) |

First run downloads MNIST / CIFAR via `torchvision` into `./data` (gitignored).

## NLP — English → French (autoregressive)

Handwritten parallel corpus in [`data.py`](src/mlscratch/nlp/en_fr_translation/data.py): 65 everyday pairs plus 205
templated sentences ("the car is white", "we like the house") that test French agreement and conjugation. Part of
the templated sentences is **held out**, so the model is scored on sentences it has never seen.
Model: small **encoder–decoder Transformer** with padding and causal masks, label smoothing, greedy autoregressive
decoding. Notes: [docs/transformer.md](docs/transformer.md).

```bash
python -m mlscratch.nlp.en_fr_translation.train   # ~2 min; prints held-out exact match and sample translations
```

```text
ok   the car is old               → la voiture est vieille       (ref: la voiture est vieille)
ok   we open the car              → nous ouvrons la voiture      (ref: nous ouvrons la voiture)
```

## What is intentionally *not* here

- Production hyperparameter sweeps
- HuggingFace wrappers for the classics
- Giant pretrained checkpoints

This repo is a **portfolio teaching kit**: open the file, understand the algorithm, run it.

## License

MIT — use freely, cite if you fork for a course.
