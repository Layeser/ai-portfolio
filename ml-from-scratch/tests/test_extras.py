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


# ------------------------------------------------------------------------------------------- phase 2 additions
from mlscratch.classical.decision_tree import DecisionTree  # noqa: E402
from mlscratch.classical.knn import KNearestNeighbors  # noqa: E402
from mlscratch.classical.linear_regression import LinearRegression  # noqa: E402
from mlscratch.classical.logistic_regression import LogisticRegression  # noqa: E402
from mlscratch.classical.naive_bayes import GaussianNaiveBayes  # noqa: E402


@pytest.fixture(scope="module")
def reg_data():
    return datasets.make_regression(n_samples=300, n_features=6, noise=10.0, random_state=0)


def test_ridge_closed_form_matches_sklearn(reg_data):
    from sklearn.linear_model import Ridge

    X, y = reg_data
    alpha = 0.1
    ours = LinearRegression(alpha=alpha).fit_closed_form(X, y)
    ref = Ridge(alpha=alpha * len(X)).fit(X, y)  # scikit-learn penalises the sum, not the mean, of squared errors
    np.testing.assert_allclose(ours.w, ref.coef_, rtol=1e-8)
    assert ours.b == pytest.approx(ref.intercept_, abs=1e-8)


def test_ridge_gradient_descent_reaches_closed_form(reg_data):
    X, y = reg_data
    gd = LinearRegression(lr=0.1, n_iters=5000, alpha=0.1).fit(X, y)
    exact = LinearRegression(alpha=0.1).fit_closed_form(X, y)
    np.testing.assert_allclose(gd.w, exact.w, rtol=1e-4)


def test_least_squares_handles_exactly_collinear_features(reg_data):
    from sklearn.linear_model import LinearRegression as SkLR

    X, y = reg_data
    Xd = np.c_[X, X[:, 0]]  # duplicated column: XᵀX is singular
    ours = LinearRegression().fit_closed_form(Xd, y)
    np.testing.assert_allclose(ours.predict(Xd), SkLR().fit(Xd, y).predict(Xd), rtol=1e-8, atol=1e-6)


def test_gradient_descent_divergence_raises(reg_data):
    X, y = reg_data
    with pytest.raises(FloatingPointError):
        LinearRegression(lr=10.0, n_iters=200).fit(X * 100, y)


def test_logistic_balanced_class_weight_raises_minority_recall():
    X, y = datasets.make_classification(n_samples=2000, weights=[0.95, 0.05], class_sep=0.8, random_state=0)
    plain = LogisticRegression(n_iters=1000).fit(X, y).predict(X)
    balanced = LogisticRegression(n_iters=1000, class_weight="balanced").fit(X, y).predict(X)
    recall = lambda pred: (pred[y == 1] == 1).mean()  # noqa: E731
    assert recall(balanced) > recall(plain)


def test_logistic_l2_shrinks_weights_on_separable_data():
    X, y = datasets.make_blobs(n_samples=200, centers=[[-2, 0], [2, 0]], cluster_std=0.5, random_state=0)
    free = LogisticRegression(n_iters=3000).fit(X, y)
    ridge = LogisticRegression(n_iters=3000, alpha=0.01).fit(X, y)
    assert np.linalg.norm(ridge.w) < np.linalg.norm(free.w)


def test_knn_distance_weights_and_batching_match_sklearn(iris_X):
    from sklearn.neighbors import KNeighborsClassifier

    y = datasets.load_iris().target
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    tr, te = idx[:100], idx[100:]
    ours = KNearestNeighbors(k=7, weights="distance", batch_size=8).fit(iris_X[tr], y[tr]).predict(iris_X[te])
    ref = KNeighborsClassifier(7, weights="distance", algorithm="brute").fit(iris_X[tr], y[tr]).predict(iris_X[te])
    assert (ours == ref).mean() >= 0.98


def test_naive_bayes_probabilities_match_sklearn(iris_X):
    from sklearn.naive_bayes import GaussianNB

    y = datasets.load_iris().target
    ours = GaussianNaiveBayes().fit(iris_X, y).predict_proba(iris_X)
    ref = GaussianNB().fit(iris_X, y).predict_proba(iris_X)
    np.testing.assert_allclose(ours, ref, atol=1e-8)


def test_decision_tree_options():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    tree = DecisionTree(max_depth=6, min_samples_leaf=10, criterion="entropy").fit(X, y)
    np.testing.assert_allclose(tree.predict_proba(X).sum(axis=1), 1.0)
    assert tree.feature_importances_.sum() == pytest.approx(1.0)

    leaf_sizes = {}
    for x in X:
        leaf = id(tree._leaf_for(x))
        leaf_sizes[leaf] = leaf_sizes.get(leaf, 0) + 1
    assert min(leaf_sizes.values()) >= 10  # min_samples_leaf is respected
    assert tree.depth() <= 6


def test_decision_tree_matches_sklearn_on_iris_training_fit(iris_X):
    from sklearn.tree import DecisionTreeClassifier

    y = datasets.load_iris().target
    ours = DecisionTree(max_depth=3).fit(iris_X, y).predict(iris_X)
    ref = DecisionTreeClassifier(max_depth=3, random_state=0).fit(iris_X, y).predict(iris_X)
    assert (ours == ref).mean() >= 0.97
