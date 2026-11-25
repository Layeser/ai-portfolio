"""Regenerate every teaching figure of docs/ in one command, deterministically.

    PYTHONPATH=src python scripts/make_figures.py

Figures go to figures/<algo>/ (versioned evidence, unlike outputs/ which is gitignored run output).
Every number quoted in docs/*.md is printed here and saved to figures/<algo>/measurements.json.
Runs on CPU in a few seconds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import KMeans as SkKMeans  # noqa: E402
from sklearn.datasets import (  # noqa: E402
    load_iris,
    make_blobs,
    make_classification,
    make_moons,
)
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from mlscratch.classical.decision_tree import DecisionTree  # noqa: E402
from mlscratch.classical.kmeans import KMeans  # noqa: E402
from mlscratch.classical.knn import KNearestNeighbors  # noqa: E402
from mlscratch.classical.linear_regression import LinearRegression  # noqa: E402
from mlscratch.classical.logistic_regression import LogisticRegression  # noqa: E402
from mlscratch.classical.naive_bayes import GaussianNaiveBayes  # noqa: E402
from mlscratch.classical.pca import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
MARKERS = ["o", "s", "^", "D"]  # one marker per class: the figures stay readable in greyscale
COLORS = ["#1f4e79", "#c55a11", "#548235", "#7030a0"]

plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11,
                     "axes.spines.top": False, "axes.spines.right": False})


def scatter_by_class(ax, Z, y, labels=None, s=18):
    for i, c in enumerate(np.unique(y)):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=s, marker=MARKERS[i % 4], facecolor="none", edgecolor=COLORS[i % 4],
                   linewidth=0.9, label=labels[i] if labels is not None else f"class {c}")


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("  wrote", path.relative_to(ROOT))


# --------------------------------------------------------------------------------------------------- PCA
def pca_figures() -> dict:
    m: dict = {}
    iris = load_iris()
    pca = PCA(2).fit(iris.data)
    Z = pca.transform(iris.data)
    r = pca.explained_variance_ratio_
    m["iris_ratio_pc1_pc2"] = [round(float(x), 4) for x in r]
    fig, ax = plt.subplots(figsize=(6, 4.4))
    scatter_by_class(ax, Z, iris.target, labels=list(iris.target_names))
    ax.set_xlabel(f"PC1 ({100 * r[0]:.1f} % of variance)")
    ax.set_ylabel(f"PC2 ({100 * r[1]:.1f} % of variance)")
    ax.set_title(f"Iris, 4 features → 2: two components keep {100 * r.sum():.1f} % of the variance")
    ax.legend(frameon=False)
    save(fig, FIG / "pca" / "iris_projection.png")

    return m


# ------------------------------------------------------------------------------------------------ k-means
def inertia(X, centroids, labels) -> float:
    return float(((X - centroids[labels]) ** 2).sum())


def kmeans_figures() -> dict:
    m: dict = {}
    X, y = make_blobs(n_samples=450, centers=3, cluster_std=0.8, random_state=0)
    km = KMeans(k=3, seed=0).fit(X)
    lab = km.labels_
    m["blobs_ari"] = round(float(adjusted_rand_score(y, lab)), 4)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    scatter_by_class(ax, X, lab, labels=[f"cluster {j}" for j in range(3)], s=14)
    ax.scatter(km.centroids[:, 0], km.centroids[:, 1], marker="X", s=180, c="black", label="centroid")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"Three round, equal-size blobs: ARI {m['blobs_ari']:.2f} against the true labels")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "kmeans" / "blobs.png")

    # initialisation: 100 seeds on 8 blobs, this implementation in three modes, against scikit-learn's best of 10
    X8, _ = make_blobs(n_samples=800, centers=8, cluster_std=0.9, random_state=4)
    ref = SkKMeans(n_clusters=8, n_init=10, random_state=0).fit(X8).inertia_
    modes = {"random_1": dict(init="random", n_init=1), "kmeanspp_1": dict(init="k-means++", n_init=1),
             "kmeanspp_10": dict(init="k-means++", n_init=10)}
    t0 = time.perf_counter()
    runs = {name: np.array([KMeans(k=8, seed=s, **kw).fit(X8).inertia_ / ref for s in range(100)])
            for name, kw in modes.items()}
    sk_random = np.array([SkKMeans(n_clusters=8, init="random", n_init=1, random_state=s).fit(X8).inertia_ / ref
                          for s in range(100)])
    sk_pp = np.array([SkKMeans(n_clusters=8, init="k-means++", n_init=1, random_state=s).fit(X8).inertia_ / ref
                      for s in range(100)])
    print(f"  (500 fits in {time.perf_counter() - t0:.1f} s, not saved: timings vary between machines)")
    m["init_reference_inertia"] = round(float(ref), 1)
    for name, r in {**runs, "sklearn_random_1": sk_random, "sklearn_kmeanspp_1": sk_pp}.items():
        m[f"init_{name}_share_more_than_1pct_worse"] = round(float(np.mean(r > 1.01)), 2)
        m[f"init_{name}_median_over_reference"] = round(float(np.median(r)), 3)
        m[f"init_{name}_worst_over_reference"] = round(float(r.max()), 2)
    bins = np.linspace(0.95, runs["random_1"].max() + 0.05, 30)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(runs["random_1"], bins=bins, color="#bfbfbf", edgecolor="black", linewidth=0.5,
            label="random init, 1 run")
    ax.hist(runs["kmeanspp_1"], bins=bins, histtype="step", color=COLORS[1], linewidth=1.8, linestyle="--",
            label="k-means++, 1 run")
    ax.hist(runs["kmeanspp_10"], bins=bins, histtype="step", color=COLORS[0], linewidth=2.2,
            label="k-means++, best of 10 runs (default)")
    ax.set_xlabel("inertia / scikit-learn best of 10 (1.0 = as good)")
    ax.set_ylabel("runs (out of 100 seeds)")
    ax.set_title(f"8 blobs, this implementation: {100 * m['init_random_1_share_more_than_1pct_worse']:.0f} % of random "
                 f"starts land in a worse minimum,\n{100 * m['init_kmeanspp_10_share_more_than_1pct_worse']:.0f} % "
                 f"with k-means++ and 10 restarts")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "kmeans" / "init_sensitivity.png")
    return m


# ------------------------------------------------------------------------------------------- shared helpers
def split(X, y, test=0.3, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    n_te = int(len(X) * test)
    return X[idx[n_te:]], X[idx[:n_te]], y[idx[n_te:]], y[idx[:n_te]]


def regions(ax, model, X, y, title, labels=None, grid=250):
    """Decision regions as light shading, points as hollow markers (readable in greyscale)."""
    pad = 0.5
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, grid),
                         np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, grid))
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    from matplotlib.colors import ListedColormap

    k = len(np.unique(y))
    shades = ListedColormap(["#d6e4f0", "#f8d9c4", "#d9ead3", "#e6d5f0"][:k])
    ax.contourf(xx, yy, zz, levels=np.arange(k + 1) - 0.5, cmap=shades)
    ax.contour(xx, yy, zz, levels=np.arange(k) + 0.5, colors="black", linewidths=0.7)
    scatter_by_class(ax, X, y, labels=labels, s=14)
    ax.set_title(title)


def acc(y, p) -> float:
    return float(np.mean(y == p))


# ------------------------------------------------------------------------------------------ linear regression
def linear_regression_figures() -> dict:
    m: dict = {}
    rng = np.random.default_rng(0)
    x = np.linspace(-1, 1, 200)
    y = 3.0 * x + 2.0 + 0.2 * rng.standard_normal(len(x))
    X = x[:, None]
    gd = LinearRegression(lr=0.1, n_iters=300).fit(X, y)
    ls = LinearRegression().fit_closed_form(X, y)
    m["toy_gd_w_b"] = [round(float(gd.w[0]), 4), round(float(gd.b), 4)]
    m["toy_ls_w_b"] = [round(float(ls.w[0]), 4), round(float(ls.b), 4)]
    ls_mse = float(np.mean((ls.predict(X) - y) ** 2))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].scatter(x, y, s=10, facecolor="none", edgecolor=COLORS[0], linewidth=0.7, label="data")
    axes[0].plot(x, ls.predict(X), color="black", linewidth=2, label=f"fit: y = {ls.w[0]:.2f} x + {ls.b:.2f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Data: y = 3x + 2 + noise")
    axes[1].plot(gd.losses, color=COLORS[0], linewidth=1.6, label="gradient descent")
    axes[1].axhline(ls_mse, color="black", linestyle="--", linewidth=1.2, label="exact least squares")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("training MSE (log)")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].set_title("Gradient descent reaches the exact solution")
    fig.suptitle(f"Recovered w = {ls.w[0]:.2f}, b = {ls.b:.2f} (true 3, 2)", fontsize=11)
    save(fig, FIG / "linear_regression" / "fit_and_convergence.png")

    # collinearity: x2 = x1 + tiny noise. Refit on 200 resamples of the noise in y.
    n = 100
    x1 = rng.standard_normal(n)
    x2 = x1 + 1e-3 * rng.standard_normal(n)
    Xc = np.c_[x1, x2]
    true = 1.0 * x1 + 1.0 * x2
    ols, rid = [], []
    for _ in range(200):
        yy = true + 0.5 * rng.standard_normal(n)
        ols.append(LinearRegression().fit_closed_form(Xc, yy).w)
        rid.append(LinearRegression(alpha=0.01).fit_closed_form(Xc, yy).w)
    ols, rid = np.array(ols), np.array(rid)
    m["collinear_ols_w1_std"] = round(float(ols[:, 0].std()), 2)
    m["collinear_ridge_w1_std"] = round(float(rid[:, 0].std()), 3)
    m["collinear_ols_sum_mean"] = round(float(ols.sum(axis=1).mean()), 3)
    m["collinear_ridge_sum_mean"] = round(float(rid.sum(axis=1).mean()), 3)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.scatter(ols[:, 0], ols[:, 1], s=12, marker="o", facecolor="none", edgecolor=COLORS[1],
               label=f"least squares (std of w1: {m['collinear_ols_w1_std']})")
    ax.scatter(rid[:, 0], rid[:, 1], s=16, marker="s", facecolor="none", edgecolor=COLORS[0],
               label=f"ridge, alpha = 0.01 (std of w1: {m['collinear_ridge_w1_std']})")
    ax.set_xlabel("w1")
    ax.set_ylabel("w2")
    ax.set_title("x2 = x1 + 0.001·noise, true w = (1, 1): least squares swings,\nonly w1 + w2 is identified; ridge stays put")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "linear_regression" / "collinearity.png")

    return m


# ---------------------------------------------------------------------------------------- logistic regression
def logistic_regression_figures() -> dict:
    m: dict = {}
    X, y = make_classification(n_samples=600, n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1,
                               class_sep=1.2, random_state=0)
    Xtr, Xte, ytr, yte = split(X, y)
    model = LogisticRegression().fit(Xtr, ytr)
    m["toy_test_accuracy"] = round(acc(yte, model.predict(Xte)), 3)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    regions(ax, model, Xte, yte, f"Linear boundary, test accuracy {m['toy_test_accuracy']:.2f}",
            labels=["class 0", "class 1"])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "logistic_regression" / "boundary.png")

    # class imbalance: 95 % / 5 %
    Xi, yi = make_classification(n_samples=4000, weights=[0.95, 0.05], class_sep=0.8, random_state=0)
    Xtr, Xte, ytr, yte = split(Xi, yi)
    res = {}
    for name, cw in (("plain", None), ("balanced", "balanced")):
        pred = LogisticRegression(n_iters=2000, class_weight=cw).fit(Xtr, ytr).predict(Xte)
        res[name] = (acc(yte, pred), float((pred[yte == 1] == 1).mean()), float((yte[pred == 1] == 1).mean()))
        m[f"imbalance_{name}_accuracy_recall_precision"] = [round(v, 3) for v in res[name]]
    m["imbalance_minority_share_test"] = round(float(yte.mean()), 3)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    xs = np.arange(3)
    for i, name in enumerate(("plain", "balanced")):
        bars = ax.bar(xs + (i - 0.5) * 0.36, res[name], 0.36, color=["#bfbfbf", COLORS[0]][i], edgecolor="black",
                      hatch=["", "//"][i], label={"plain": "no weights", "balanced": 'class_weight="balanced"'}[name])
        ax.bar_label(bars, [f"{v:.2f}" for v in res[name]], fontsize=8)
    ax.set_xticks(xs, ["accuracy", "recall, minority class", "precision, minority class"])
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{100 * m['imbalance_minority_share_test']:.0f} % positives: accuracy looks fine "
                 f"while most positives are missed")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "logistic_regression" / "imbalance.png")
    return m


# ------------------------------------------------------------------------------------------------------- k-NN
def knn_figures() -> dict:
    m: dict = {}
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=0)
    Xtr, Xte, ytr, yte = split(X, y)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, k in zip(axes, (1, 15), strict=True):
        mdl = KNearestNeighbors(k=k).fit(Xtr, ytr)
        tr, te = acc(ytr, mdl.predict(Xtr)), acc(yte, mdl.predict(Xte))
        m[f"moons_k{k}_train_test"] = [round(tr, 3), round(te, 3)]
        regions(ax, mdl, Xtr, ytr, f"k = {k}: train {tr:.2f}, test {te:.2f}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
    fig.suptitle("k controls the smoothness: k = 1 memorises the noise (train 1.00), k = 15 does better on test",
                 fontsize=11)
    save(fig, FIG / "knn" / "k_smoothness.png")

    return m


# ---------------------------------------------------------------------------------------------- naive Bayes
def naive_bayes_figures() -> dict:
    m: dict = {}
    iris = load_iris()
    X2 = iris.data[:, 2:4]
    Xtr, Xte, ytr, yte = split(X2, iris.target)
    nb = GaussianNaiveBayes().fit(Xtr, ytr)
    m["iris_petals_test_accuracy"] = round(acc(yte, nb.predict(Xte)), 3)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    regions(ax, nb, X2, iris.target, f"Iris, petal features: test accuracy {m['iris_petals_test_accuracy']:.2f}",
            labels=list(iris.target_names))
    ax.set_xlabel("petal length (cm)")
    ax.set_ylabel("petal width (cm)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "naive_bayes" / "iris_regions.png")

    return m


# -------------------------------------------------------------------------------------------- decision tree
def decision_tree_figures() -> dict:
    m: dict = {}
    iris = load_iris()
    X2 = iris.data[:, 2:4]
    Xtr, Xte, ytr, yte = split(X2, iris.target)
    tree = DecisionTree(max_depth=3).fit(Xtr, ytr)
    m["iris_petals_depth3_test_accuracy"] = round(acc(yte, tree.predict(Xte)), 3)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    regions(ax, tree, X2, iris.target, f"Depth 3 on petal features: test accuracy "
            f"{m['iris_petals_depth3_test_accuracy']:.2f}", labels=list(iris.target_names))
    ax.set_xlabel("petal length (cm)")
    ax.set_ylabel("petal width (cm)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "decision_tree" / "iris_regions.png")

    # depth against train / test accuracy on noisy data
    Xm, ym = make_moons(n_samples=600, noise=0.35, random_state=0)
    Xtr, Xte, ytr, yte = split(Xm, ym)
    depths = list(range(1, 21))
    tr_acc, te_acc = [], []
    for dpt in depths:
        t = DecisionTree(max_depth=dpt).fit(Xtr, ytr)
        tr_acc.append(acc(ytr, t.predict(Xtr)))
        te_acc.append(acc(yte, t.predict(Xte)))
    best = int(np.argmax(te_acc))
    m["depth_best_test"] = {"depth": depths[best], "test": round(te_acc[best], 3), "train": round(tr_acc[best], 3)}
    m["depth_20"] = {"train": round(tr_acc[-1], 3), "test": round(te_acc[-1], 3)}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(depths, tr_acc, "o-", color=COLORS[0], linewidth=1.6, markersize=4, label="train")
    ax.plot(depths, te_acc, "s--", color=COLORS[1], linewidth=1.6, markersize=4, label="test")
    ax.set_xlabel("max_depth")
    ax.set_xticks(depths[::2])
    ax.set_ylabel("accuracy")
    ax.set_title(f"Noisy moons: train accuracy reaches {tr_acc[-1]:.2f}, test peaks at {te_acc[best]:.2f} "
                 f"(depth {depths[best]})\nthen falls to {te_acc[-1]:.2f}: the tree memorises the noise")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "decision_tree" / "depth_overfitting.png")

    return m


def main() -> None:
    np.random.seed(0)
    t0 = time.perf_counter()
    for name, fn in (("pca", pca_figures), ("kmeans", kmeans_figures),
                     ("linear_regression", linear_regression_figures),
                     ("logistic_regression", logistic_regression_figures), ("knn", knn_figures),
                     ("naive_bayes", naive_bayes_figures), ("decision_tree", decision_tree_figures)):
        print(name)
        m = fn()
        (FIG / name).mkdir(parents=True, exist_ok=True)
        (FIG / name / "measurements.json").write_text(json.dumps(m, indent=2) + "\n")
        for k, v in m.items():
            print(f"  {k}: {v}")
    print(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
