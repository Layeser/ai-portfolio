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
from sklearn.datasets import load_iris, load_wine, make_blobs, make_circles, make_moons  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from mlscratch.classical.kmeans import KMeans  # noqa: E402
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


def main() -> None:
    np.random.seed(0)
    t0 = time.perf_counter()
    for name, fn in (("pca", pca_figures), ("kmeans", kmeans_figures)):
        print(name)
        m = fn()
        (FIG / name).mkdir(parents=True, exist_ok=True)
        (FIG / name / "measurements.json").write_text(json.dumps(m, indent=2) + "\n")
        for k, v in m.items():
            print(f"  {k}: {v}")
    print(f"done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
