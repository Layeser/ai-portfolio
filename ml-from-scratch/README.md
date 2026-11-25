# ML from Scratch

Implementations you can **read in one sitting**. No black-box `sklearn.fit`.  
NumPy for classical ML. PyTorch only where tensors / autograd earn their keep.

Inspired by the teaching style of [Andrej Karpathy](https://karpathy.ai/) — small models, clear math, runnable demos.

Each algorithm gets a one-page note in [`docs/`](docs/README.md): the maths as implemented, where it breaks (with
measured figures), and the gap with a production library.

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
PYTHONPATH=src python scripts/make_figures.py   # regenerates figures/<algo>/*.png and measurements.json, deterministic, a few seconds on CPU
```

## Correctness: checked against scikit-learn

Writing an algorithm from the maths only counts if it gives the right answer. `tests/` (13 tests, ~15 s on CPU,
run in CI) compares every classical model with its scikit-learn reference, with explicit tolerances:

| Model | Check |
|---|---|
| Linear regression | closed form = sklearn coefficients (rtol 1e-6); gradient descent converges to the closed form |
| PCA | explained variance identical; components identical up to sign |
| k-NN, Gaussian naive Bayes | ≥ 97 % identical predictions (only distance ties may differ) |
| Logistic regression, decision tree | test accuracy within 2 / 5 points of sklearn on breast cancer |
| k-means | ARI > 0.95 on separated blobs; inertia within 5 % of sklearn's |
| LeNet, ResNet-18, autoencoder, VAE | output shapes; KL term = 0 for a standard-normal posterior; each model overfits a tiny batch (classic bug detector) |

```bash
pip install -r requirements.txt
pytest tests -q
```

## Classical ML (NumPy)

| Algorithm | Dataset | Run | Notes |
|-----------|---------|-----|-------|
| Linear regression | `make_regression` | `python -m mlscratch.classical.linear_regression` | — |
| Logistic regression | `make_classification` | `python -m mlscratch.classical.logistic_regression` | — |
| k-NN | Iris | `python -m mlscratch.classical.knn` | — |
| Gaussian Naive Bayes | Iris | `python -m mlscratch.classical.naive_bayes` | — |
| Decision tree (CART, Gini) | Iris | `python -m mlscratch.classical.decision_tree` | — |
| k-Means | blobs | `python -m mlscratch.classical.kmeans` | [page](docs/kmeans.md) |
| PCA | Iris (2D viz) | `python -m mlscratch.classical.pca` | [page](docs/pca.md) |

Plots land in `outputs/<algo>/`.

## Deep Learning (PyTorch)

| Model | Task | Run | Notes |
|-------|------|-----|-------|
| Autoencoder | MNIST reconstruction | `python -m mlscratch.deep_learning.autoencoder` | — |
| VAE | MNIST + latent walk | `python -m mlscratch.deep_learning.vae` | — |
| LeNet-5 | MNIST classification | `python -m mlscratch.deep_learning.lenet` | — |
| ResNet-18 (CIFAR variant) | CIFAR-10 | `python -m mlscratch.deep_learning.resnet` | — |

First run downloads MNIST / CIFAR via `torchvision` into `./data` (gitignored).

## NLP — English → French (autoregressive)

Handwritten parallel corpus in [`data.py`](src/mlscratch/nlp/en_fr_translation/data.py) (~65 pairs).  
Model: small **Encoder–Decoder Transformer** with causal masking; inference is **greedy autoregressive** token-by-token.

```bash
python -m mlscratch.nlp.en_fr_translation.train
```

You should see training loss drop and sample translations like:

```text
EN: i love machine learning
FR: j aime l apprentissage automatique
```

(Exact strings depend on seed / epochs; the point is the full train → generate loop.)

## What is intentionally *not* here

- Production hyperparameter sweeps
- HuggingFace wrappers for the classics
- Giant pretrained checkpoints

This repo is a **portfolio teaching kit**: open the file, understand the algorithm, run it.

## License

MIT — use freely, cite if you fork for a course.
