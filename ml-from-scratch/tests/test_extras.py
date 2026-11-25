"""Tests for what the implementations add beyond the textbook core (solvers, signs, inverse, init, restarts)."""

import warnings

import numpy as np
import pytest
from sklearn import datasets
from sklearn.cluster import KMeans as SkKMeans

from mlscratch.classical.kmeans import KMeans
from mlscratch.classical.pca import PCA


@pytest.fixture(scope="module")
def iris_X():
    return datasets.load_iris().data


def test_pca_eigh_and_svd_solvers_agree(iris_X):
    a, b = PCA(3, solver="eigh").fit(iris_X), PCA(3, solver="svd").fit(iris_X)
    np.testing.assert_allclose(a.explained_variance_ratio_, b.explained_variance_ratio_, rtol=1e-10)
    np.testing.assert_allclose(a.components_, b.components_, atol=1e-8)  # same signs thanks to the sign convention


def test_pca_sign_is_deterministic_under_row_permutation(iris_X):
    perm = np.random.default_rng(0).permutation(len(iris_X))
    np.testing.assert_allclose(PCA(2).fit(iris_X).components_, PCA(2).fit(iris_X[perm]).components_, atol=1e-10)


def test_pca_inverse_transform_and_whitening(iris_X):
    full = PCA(4).fit(iris_X)
    np.testing.assert_allclose(full.inverse_transform(full.transform(iris_X)), iris_X, atol=1e-10)
    white = PCA(2, whiten=True).fit(iris_X)
    np.testing.assert_allclose(white.transform(iris_X).var(axis=0, ddof=1), 1.0, rtol=1e-10)
    np.testing.assert_allclose(white.inverse_transform(white.transform(iris_X)),
                               PCA(2).fit(iris_X).inverse_transform(PCA(2).fit_transform(iris_X)), atol=1e-10)


def test_pca_rejects_bad_input(iris_X):
    with pytest.raises(ValueError):
        PCA(5).fit(iris_X)  # more components than features
    bad = iris_X.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        PCA(2).fit(bad)


def test_kmeanspp_with_restarts_matches_sklearn_on_hard_case():
    X, _ = datasets.make_blobs(n_samples=800, centers=8, cluster_std=0.9, random_state=4)
    ours = KMeans(k=8, seed=0).fit(X)  # defaults: k-means++, 10 restarts
    ref = SkKMeans(n_clusters=8, n_init=10, random_state=0).fit(X)
    assert ours.inertia_ <= ref.inertia_ * 1.01
    np.testing.assert_array_equal(ours.labels_, ours.predict(X))


def test_kmeans_single_random_start_is_worse_than_kmeanspp_on_average():
    X, _ = datasets.make_blobs(n_samples=800, centers=8, cluster_std=0.9, random_state=4)
    rnd = [KMeans(k=8, seed=s, init="random", n_init=1).fit(X).inertia_ for s in range(20)]
    pp = [KMeans(k=8, seed=s, init="k-means++", n_init=1).fit(X).inertia_ for s in range(20)]
    assert np.median(pp) < np.median(rnd)


def test_kmeans_converges_without_warning_on_easy_data():
    X, _ = datasets.make_blobs(n_samples=300, centers=3, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        KMeans(k=3).fit(X)
