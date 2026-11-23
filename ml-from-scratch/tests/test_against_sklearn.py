"""Correctness tests: every from-scratch model is checked against the scikit-learn reference.

Writing an algorithm from the maths is only convincing if it gives the same answer as a
battle-tested implementation. Tolerances are explicit: exact match where the algorithm is
deterministic (PCA, closed-form regression), close accuracy where optimisation differs.
"""


import numpy as np
import pytest
from mlscratch.classical.decision_tree import DecisionTree  # noqa: E402
from mlscratch.classical.kmeans import KMeans  # noqa: E402
from mlscratch.classical.knn import KNearestNeighbors  # noqa: E402
from mlscratch.classical.linear_regression import LinearRegression  # noqa: E402
from mlscratch.classical.logistic_regression import LogisticRegression  # noqa: E402
from mlscratch.classical.naive_bayes import GaussianNaiveBayes  # noqa: E402
from mlscratch.classical.pca import PCA  # noqa: E402
from sklearn import datasets
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@pytest.fixture(scope="module")
def regression():
    X, y = datasets.make_regression(n_samples=400, n_features=5, noise=5.0, random_state=0)
    return X, y


@pytest.fixture(scope="module")
def cancer():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    return scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te


@pytest.fixture(scope="module")
def iris():
    X, y = datasets.load_iris(return_X_y=True)
    return train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


def test_linear_regression_closed_form_matches_sklearn(regression):
    from sklearn.linear_model import LinearRegression as SkLR

    X, y = regression
    ours = LinearRegression().fit_closed_form(X, y)
    ref = SkLR().fit(X, y)
    np.testing.assert_allclose(ours.w, ref.coef_, rtol=1e-6)
    assert ours.b == pytest.approx(ref.intercept_, abs=1e-6)


def test_linear_regression_gradient_descent_converges_to_closed_form(regression):
    X, y = regression
    gd = LinearRegression(lr=0.1, n_iters=3000).fit(X, y)
    exact = LinearRegression().fit_closed_form(X, y)
    np.testing.assert_allclose(gd.w, exact.w, rtol=1e-3)
    assert gd.losses[-1] < gd.losses[0]


def test_logistic_regression_accuracy_close_to_sklearn(cancer):
    from sklearn.linear_model import LogisticRegression as SkLogit

    X_tr, X_te, y_tr, y_te = cancer
    ours = accuracy_score(y_te, LogisticRegression(lr=0.1, n_iters=3000).fit(X_tr, y_tr).predict(X_te))
    ref = accuracy_score(y_te, SkLogit(max_iter=5000).fit(X_tr, y_tr).predict(X_te))
    assert ours >= ref - 0.02


def test_knn_predictions_match_sklearn(iris):
    from sklearn.neighbors import KNeighborsClassifier

    X_tr, X_te, y_tr, y_te = iris
    ours = KNearestNeighbors(k=5).fit(X_tr, y_tr).predict(X_te)
    ref = KNeighborsClassifier(n_neighbors=5, algorithm="brute").fit(X_tr, y_tr).predict(X_te)
    assert (ours == ref).mean() >= 0.97  # only distance ties may break differently


def test_gaussian_naive_bayes_matches_sklearn(iris):
    from sklearn.naive_bayes import GaussianNB

    X_tr, X_te, y_tr, y_te = iris
    ours = GaussianNaiveBayes().fit(X_tr, y_tr).predict(X_te)
    ref = GaussianNB().fit(X_tr, y_tr).predict(X_te)
    assert (ours == ref).mean() >= 0.97


def test_decision_tree_accuracy_close_to_sklearn(cancer):
    from sklearn.tree import DecisionTreeClassifier

    X_tr, X_te, y_tr, y_te = cancer
    ours = accuracy_score(y_te, DecisionTree(max_depth=5).fit(X_tr, y_tr).predict(X_te))
    ref = accuracy_score(y_te, DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_tr, y_tr).predict(X_te))
    assert ours >= ref - 0.05


def test_kmeans_recovers_well_separated_clusters():
    from sklearn.cluster import KMeans as SkKMeans

    X, y = datasets.make_blobs(n_samples=600, centers=4, cluster_std=0.8, random_state=3)
    ours = KMeans(k=4, seed=0).fit(X)
    ref = SkKMeans(n_clusters=4, n_init=10, random_state=0).fit(X)
    assert adjusted_rand_score(y, ours.predict(X)) > 0.95
    inertia = ((X - ours.centroids[ours.predict(X)]) ** 2).sum()
    assert inertia <= ref.inertia_ * 1.05


def test_pca_matches_sklearn_up_to_sign(iris):
    from sklearn.decomposition import PCA as SkPCA

    X = np.vstack(iris[:2])
    ours = PCA(n_components=2).fit(X)
    ref = SkPCA(n_components=2).fit(X)
    np.testing.assert_allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-6)
    for ours_c, ref_c in zip(ours.components_.T, ref.components_, strict=True):
        assert abs(abs(ours_c @ ref_c) - 1) < 1e-6  # same direction, sign is arbitrary
