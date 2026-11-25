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
    load_breast_cancer,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
)
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from mlscratch.classical.decision_tree import DecisionTree  # noqa: E402
from mlscratch.classical.kmeans import KMeans  # noqa: E402
from mlscratch.classical.knn import KNearestNeighbors  # noqa: E402
from mlscratch.classical.linear_regression import LinearRegression  # noqa: E402
from mlscratch.classical.logistic_regression import LogisticRegression, sigmoid  # noqa: E402
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

    # failure 1: unscaled features. Wine: proline is in the hundreds/thousands, hue around 1.
    wine = load_wine()
    X = wine.data
    raw = PCA(2).fit(X)
    std = (X - X.mean(0)) / X.std(0, ddof=1)
    scaled = PCA(2).fit(std)
    top = int(np.argmax(np.abs(raw.components_[:, 0])))
    m["wine_raw_ratio_pc1"] = round(float(raw.explained_variance_ratio_[0]), 4)
    m["wine_raw_pc1_top_feature"] = wine.feature_names[top]
    m["wine_raw_pc1_top_loading"] = round(float(abs(raw.components_[top, 0])), 4)
    m["wine_scaled_ratio_pc1_pc2"] = [round(float(x), 4) for x in scaled.explained_variance_ratio_]
    m["wine_proline_std"] = round(float(X[:, top].std(ddof=1)), 1)
    m["wine_hue_std"] = round(float(X[:, wine.feature_names.index("hue")].std(ddof=1)), 3)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    scatter_by_class(axes[0], raw.transform(X), wine.target, labels=list(wine.target_names))
    axes[0].set_title(f"Raw units: PC1 = {wine.feature_names[top]} ({100 * raw.explained_variance_ratio_[0]:.1f} %)")
    scatter_by_class(axes[1], scaled.transform(std), wine.target, labels=list(wine.target_names))
    axes[1].set_title(f"Standardised: PC1 {100 * scaled.explained_variance_ratio_[0]:.1f} %, "
                      f"PC2 {100 * scaled.explained_variance_ratio_[1]:.1f} %")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Wine without standardisation: proline (std {m['wine_proline_std']:.0f}) outweighs hue "
                 f"(std {m['wine_hue_std']:.2f}) and decides PC1", fontsize=11)
    save(fig, FIG / "pca" / "wine_unscaled_vs_standardised.png")

    # failure 2: structure that is not linear. Two concentric circles; PCA to 1D mixes them completely.
    Xc, yc = make_circles(n_samples=400, factor=0.4, noise=0.05, random_state=0)
    p1 = PCA(1).fit(Xc)
    z = p1.transform(Xc)[:, 0]
    m["circles_ratio_pc1"] = round(float(p1.explained_variance_ratio_[0]), 4)
    # best single threshold on PC1, as a measure of how separable the classes are after projection
    zs = np.sort(z)
    best = max(max(np.mean((z > t) == yc), np.mean((z <= t) == yc)) for t in (zs[:-1] + zs[1:]) / 2)
    m["circles_best_threshold_accuracy_on_pc1"] = round(float(best), 4)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1, 1.3]})
    scatter_by_class(axes[0], Xc, yc, labels=["outer", "inner"], s=12)
    axes[0].set_aspect("equal")
    axes[0].set_title("Input: two concentric circles")
    axes[0].legend(frameon=False, fontsize=8)
    for i, c in enumerate([0, 1]):
        axes[1].hist(z[yc == c], bins=30, histtype="step", color=COLORS[i], linewidth=1.6,
                     linestyle=["-", "--"][i], label=["outer", "inner"][i])
    axes[1].set_xlabel("position on PC1")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"After PCA to 1-D: best threshold separates {100 * best:.0f} %")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("A linear projection cannot unfold a non-linear structure", fontsize=11)
    save(fig, FIG / "pca" / "circles_nonlinear.png")
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

    # failure 1: non-convex clusters
    Xm, ym = make_moons(n_samples=400, noise=0.06, random_state=0)
    kmm = KMeans(k=2, seed=0).fit(Xm)
    labm = kmm.predict(Xm)
    m["moons_ari"] = round(float(adjusted_rand_score(ym, labm)), 4)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    scatter_by_class(axes[0], Xm, ym, labels=["moon A", "moon B"], s=12)
    axes[0].set_title("True structure: two crescents")
    scatter_by_class(axes[1], Xm, labm, labels=["cluster 0", "cluster 1"], s=12)
    axes[1].scatter(kmm.centroids[:, 0], kmm.centroids[:, 1], marker="X", s=160, c="black")
    axes[1].set_title(f"k-means, k = 2: ARI {m['moons_ari']:.2f}")
    for ax in axes:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("k-means cuts the plane with a straight line: crescents become two halves", fontsize=11)
    save(fig, FIG / "kmeans" / "moons.png")

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

    # gradient descent and feature scale: one feature in [-1, 1], one multiplied by 100
    Xs = np.c_[rng.uniform(-1, 1, 300), rng.uniform(-1, 1, 300)]
    ys = Xs @ np.array([2.0, -1.0]) + 0.1 * rng.standard_normal(300)
    Xu = Xs * np.array([1.0, 100.0])
    target = float(np.mean((LinearRegression().fit_closed_form(Xs, ys).predict(Xs) - ys) ** 2)) * 1.01

    def iters_to(Xa, lr):
        r = LinearRegression(lr=lr, n_iters=20000).fit(Xa, ys)
        hit = np.nonzero(np.array(r.losses) <= target)[0]
        return (int(hit[0]) + 1 if len(hit) else None), r.losses

    lam = np.linalg.eigvalsh(Xu.T @ Xu / len(Xu)).max()
    lr_u = 1.0 / lam  # largest step that keeps the unscaled problem stable, up to a factor 2
    it_s, loss_s = iters_to(Xs, 0.5)
    it_u, loss_u = iters_to(Xu, lr_u)
    m["scale_iters_standard_lr0.5"] = it_s
    m["scale_iters_unscaled_max_stable_lr"] = it_u
    m["scale_unscaled_lr"] = float(f"{lr_u:.2e}")
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            LinearRegression(lr=0.5, n_iters=200).fit(Xu, ys)
        m["scale_unscaled_lr0.5_diverges"] = False
    except FloatingPointError:
        m["scale_unscaled_lr0.5_diverges"] = True
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(loss_s[:3000], color=COLORS[0], linewidth=1.8, label="features in [-1, 1], lr = 0.5")
    ax.plot(loss_u[:3000], color=COLORS[1], linewidth=1.8, linestyle="--",
            label=f"second feature × 100, lr = {lr_u:.1e} (largest stable)")
    ax.axhline(target, color="black", linewidth=0.8, linestyle=":")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("training MSE (log)")
    ax.set_title(f"Same data, one feature in other units: {it_s} iterations vs {it_u or '> 20000'}\n"
                 f"(lr = 0.5 on the unscaled data diverges)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "linear_regression" / "feature_scale.png")
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

    # separable data: without a penalty, ‖w‖ keeps growing
    Xs, ys = make_blobs(n_samples=200, centers=[[-2, 0], [2, 0]], cluster_std=0.5, random_state=0)
    norms = {}
    for alpha in (0.0, 0.01):
        # the same update as LogisticRegression.fit, unrolled to record ‖w‖ at every step
        path, w, b = [], np.zeros(2), 0.0
        for _ in range(20000):
            p = sigmoid(Xs @ w + b)
            err = p - ys
            w = w - 0.5 * ((Xs.T @ err) / len(ys) + alpha * w)
            b = b - 0.5 * np.mean(err)
            path.append(np.linalg.norm(w))
        norms[alpha] = np.array(path)
    m["separable_norm_no_penalty_at_1k_10k_20k"] = [round(float(norms[0.0][i]), 2) for i in (999, 9999, 19999)]
    m["separable_norm_l2_0.01_at_1k_10k_20k"] = [round(float(norms[0.01][i]), 2) for i in (999, 9999, 19999)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(np.arange(1, 20001), norms[0.0], color=COLORS[1], linewidth=1.8, linestyle="--", label="no penalty")
    ax.plot(np.arange(1, 20001), norms[0.01], color=COLORS[0], linewidth=1.8, label="L2, alpha = 0.01")
    ax.set_xscale("log")
    ax.set_xlabel("iteration (log)")
    ax.set_ylabel("‖w‖")
    ax.set_title("Perfectly separable data: without a penalty ‖w‖ never stops growing")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "logistic_regression" / "separable_weights.png")

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

    # query time against n (brute force: every query scans every training point)
    rng = np.random.default_rng(0)
    sizes = [1_000, 5_000, 20_000, 50_000, 100_000, 200_000]
    Q = rng.standard_normal((200, 10))
    times = []
    for n in sizes:
        mdl = KNearestNeighbors(k=5).fit(rng.standard_normal((n, 10)), rng.integers(0, 2, n))
        mdl.predict(Q[:10])  # warm-up
        t0 = time.perf_counter()
        mdl.predict(Q)
        times.append((time.perf_counter() - t0) / len(Q) * 1e3)
    m["timing_note"] = "milliseconds per query, d = 10, k = 5, this machine; ratios are the meaningful part"
    m["timing_ms_per_query"] = {str(n): round(t, 3) for n, t in zip(sizes, times, strict=True)}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(sizes, times, "o-", color=COLORS[0], linewidth=1.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training points n (log)")
    ax.set_ylabel("time per query, ms (log)")
    ax.set_title("No index: every query scans all n points,\nso the time per query grows with n (timings vary by machine)")
    save(fig, FIG / "knn" / "query_time.png")

    # irrelevant features: 2 informative dimensions + noise dimensions
    Xb, yb = make_moons(n_samples=600, noise=0.2, random_state=1)
    extra = [0, 2, 5, 10, 20, 50]
    scores = []
    for e in extra:
        Xe = np.c_[Xb, np.random.default_rng(e).standard_normal((len(Xb), e))] if e else Xb
        Xe = (Xe - Xe.mean(0)) / Xe.std(0)
        a, b_, c, d_ = split(Xe, yb)
        scores.append(acc(d_, KNearestNeighbors(k=15).fit(a, c).predict(b_)))
    m["irrelevant_features_test_accuracy"] = {str(e): round(s, 3) for e, s in zip(extra, scores, strict=True)}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(extra, scores, "s-", color=COLORS[1], linewidth=1.6)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1, label="chance")
    ax.set_xlabel("irrelevant (noise) features added to the 2 informative ones")
    ax.set_ylabel("test accuracy, k = 15")
    ax.set_ylim(0.4, 1.02)
    ax.set_title(f"Distances drown in irrelevant dimensions: {scores[0]:.2f} → {scores[-1]:.2f}")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "knn" / "irrelevant_features.png")
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

    # correlated features: duplicate the features r times; the evidence is counted r times
    Xc, yc = load_breast_cancer(return_X_y=True)
    Xc = (Xc - Xc.mean(0)) / Xc.std(0)
    Xtr, Xte, ytr, yte = split(Xc, yc)
    copies = [1, 2, 4, 8]
    ll, ac, conf = [], [], []
    for r in copies:
        mdl = GaussianNaiveBayes().fit(np.tile(Xtr, r), ytr)
        p = mdl.predict_proba(np.tile(Xte, r))
        ll.append(float(-np.mean(np.log(np.clip(p[np.arange(len(yte)), yte], 1e-15, 1)))))
        ac.append(acc(yte, mdl.classes_[p.argmax(1)]))
        conf.append(float(np.mean(p.max(1) > 0.999)))
    m["duplicates_copies"] = copies
    m["duplicates_log_loss"] = [round(v, 3) for v in ll]
    m["duplicates_accuracy"] = [round(v, 3) for v in ac]
    m["duplicates_share_predictions_above_0.999"] = [round(v, 3) for v in conf]
    fig, ax1 = plt.subplots(figsize=(6.5, 4))
    ax1.plot(copies, ll, "o-", color=COLORS[1], linewidth=1.6, label="log loss (lower is better)")
    ax1.plot(copies, ac, "s--", color=COLORS[0], linewidth=1.6, label="accuracy")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(copies, [str(c) for c in copies])
    ax1.set_xlabel("copies of every feature (perfectly correlated duplicates)")
    ax1.set_title(f"Duplicated features: accuracy stays {ac[0]:.2f}, log loss {ll[0]:.2f} → {ll[-1]:.2f}\n"
                  f"(the same evidence counted {copies[-1]} times: overconfidence)")
    ax1.legend(frameon=False, fontsize=8)
    save(fig, FIG / "naive_bayes" / "correlated_features.png")

    # a class made of two blobs (XOR): one Gaussian per class cannot fit it
    rng = np.random.default_rng(0)
    centres = np.array([[-2, -2], [2, 2], [-2, 2], [2, -2]])
    Xx = np.vstack([c + 0.7 * rng.standard_normal((150, 2)) for c in centres])
    yx = np.repeat([0, 0, 1, 1], 150)
    a, b_, c, d_ = split(Xx, yx)
    nbx = GaussianNaiveBayes().fit(a, c)
    m["xor_nb_test_accuracy"] = round(acc(d_, nbx.predict(b_)), 3)
    m["xor_knn15_test_accuracy"] = round(acc(d_, KNearestNeighbors(k=15).fit(a, c).predict(b_)), 3)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    regions(ax, nbx, Xx, yx, f"Each class is two blobs: naive Bayes {m['xor_nb_test_accuracy']:.2f} "
            f"(k-NN: {m['xor_knn15_test_accuracy']:.2f})", labels=["class 0", "class 1"])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "naive_bayes" / "multimodal_class.png")
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

    # a diagonal boundary: axis-aligned splits can only approximate it with a staircase
    rng = np.random.default_rng(0)
    Xd = rng.uniform(-1, 1, (600, 2))
    yd = (Xd[:, 1] > Xd[:, 0]).astype(int)
    a, b_, c, d_ = split(Xd, yd)
    res = {}
    for dpt in (2, 4, 8):
        res[dpt] = acc(d_, DecisionTree(max_depth=dpt).fit(a, c).predict(b_))
    m["diagonal_test_accuracy_by_depth"] = {str(k): round(v, 3) for k, v in res.items()}
    m["diagonal_logistic_test_accuracy"] = round(acc(d_, LogisticRegression(n_iters=3000).fit(a, c).predict(b_)), 3)
    t4 = DecisionTree(max_depth=4).fit(a, c)
    fig, ax = plt.subplots(figsize=(6, 4.4))
    regions(ax, t4, Xd, yd, f"Boundary y = x, depth 4: a staircase, test {res[4]:.2f}\n"
            f"(logistic regression: {m['diagonal_logistic_test_accuracy']:.2f})", labels=["below", "above"])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(frameon=False, fontsize=8)
    save(fig, FIG / "decision_tree" / "diagonal_staircase.png")
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
