"""
test_ds.py

Pytest-based tests for HW-05: Clustering Techniques on Energy Data.

These tests assume the following functions are defined in ds.py:

- load_energy_data
- initial_summary
- clean_energy_data
- filter_year_range
- select_clustering_features
- standardize_features
- compute_calinski_harabasz_scores
- run_kmeans
- run_hierarchical_clustering
- fit_gmm
- gmm_cluster_labels
- gmm_outlier_mask
- silhouette_for_labels
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import ds


# ============================================================
# Helpers
# ============================================================

def _make_energy_like_df() -> pd.DataFrame:
    """Create a tiny synthetic DataFrame mimicking energy data."""
    return pd.DataFrame(
        {
            "country": ["A", "B", "A", "C"],
            "year": [2000, 2000, 2010, 2005],
            "fossil_energy_per_capita": [1.0, 2.0, 1.2, 0.5],
            "renewables_energy_per_capita": [0.2, 0.1, 0.3, 0.8],
            "low_carbon_share_energy": [0.1, 0.05, 0.2, 0.6],
        }
    )


# ============================================================
# Part 1: Data Preparation & Feature Selection
# ============================================================

def test_load_energy_data(tmp_path):
    """load_energy_data should read CSV into a DataFrame."""
    csv_path = tmp_path / "energy_small.csv"
    df_expected = pd.DataFrame(
        {
            "country": ["A", "B"],
            "year": [2000, 2005],
            "fossil_energy_per_capita": [1.0, 0.5],
        }
    )
    df_expected.to_csv(csv_path, index=False)

    df_loaded = ds.load_energy_data(str(csv_path))

    assert isinstance(df_loaded, pd.DataFrame)
    pd.testing.assert_frame_equal(df_loaded, df_expected)


def test_initial_summary():
    """initial_summary should correctly separate numeric and categorical columns."""
    df = _make_energy_like_df()
    summary = ds.initial_summary(df)

    assert summary["n_rows"] == len(df)
    assert summary["n_cols"] == df.shape[1]

    cols = summary["columns"]
    assert isinstance(cols, list)
    assert set(cols) == set(df.columns)

    numeric_cols = summary["numeric_columns"]
    cat_cols = summary["categorical_columns"]

    # Basic sanity checks
    assert "year" in numeric_cols
    assert "fossil_energy_per_capita" in numeric_cols
    assert "country" in cat_cols
    # No overlap
    assert set(numeric_cols).isdisjoint(set(cat_cols))


def test_clean_energy_data_drops_duplicates_and_missing():
    """clean_energy_data should drop duplicates and rows with missing subset columns."""
    df = pd.DataFrame(
        {
            "country": ["A", "A", "B", "C"],
            "year": [2000, 2000, 2005, 2010],
            "fossil_energy_per_capita": [1.0, 1.0, np.nan, 0.3],
            "renewables_energy_per_capita": [0.2, 0.2, 0.1, np.nan],
        }
    )
    # Introduce a duplicate row (0 and 1)
    subset_cols = ["fossil_energy_per_capita", "renewables_energy_per_capita"]
    cleaned = ds.clean_energy_data(df, subset_cols=subset_cols)

    # No duplicates
    assert len(cleaned) < len(df)
    # No missing in subset_cols
    assert not cleaned[subset_cols].isna().any().any()
    # Original unchanged
    assert df.isna().any().any()


def test_filter_year_range():
    """filter_year_range should keep only rows with year in [start, end]."""
    df = _make_energy_like_df()
    filtered = ds.filter_year_range(df, 2000, 2005)

    assert filtered["year"].min() >= 2000
    assert filtered["year"].max() <= 2005
    # At least one row dropped
    assert len(filtered) < len(df)
    # Original unchanged
    assert len(df) == 4


def test_select_clustering_features():
    """select_clustering_features should return a narrow feature DataFrame."""
    df = _make_energy_like_df()
    feature_cols = ["fossil_energy_per_capita", "renewables_energy_per_capita"]
    feats = ds.select_clustering_features(df, feature_cols)

    assert list(feats.columns) == feature_cols
    assert feats.shape[0] == len(df)
    # Original unchanged
    assert set(df.columns) != set(feats.columns)


def test_standardize_features_mean_zero_std_one():
    """standardize_features should produce columns with mean≈0 and std≈1."""
    df = _make_energy_like_df()
    feature_cols = [
        "fossil_energy_per_capita",
        "renewables_energy_per_capita",
        "low_carbon_share_energy",
    ]
    feats = df[feature_cols]

    df_scaled, means, stds = ds.standardize_features(feats)

    assert isinstance(df_scaled, pd.DataFrame)
    assert list(df_scaled.columns) == feature_cols
    assert df_scaled.shape == feats.shape

    # Means approx 0, stds approx 1
    col_means = df_scaled.mean().values
    col_stds = df_scaled.std(ddof=0).values  # population std

    assert np.allclose(col_means, np.zeros_like(col_means), atol=1e-7)
    assert np.allclose(col_stds, np.ones_like(col_stds), atol=1e-7)

    # Returned means/stds should match original
    assert np.allclose(means, feats.mean().values)
    assert np.allclose(stds, feats.std(ddof=0).values)


# ============================================================
# Part 2: K-Means & Hierarchical Clustering
# ============================================================

def test_compute_calinski_harabasz_scores():
    """compute_calinski_harabasz_scores should match sklearn's CH scores."""
    # Simple 2D blob data
    rng = np.random.default_rng(0)
    X1 = rng.normal(loc=[0.0, 0.0], scale=0.2, size=(20, 2))
    X2 = rng.normal(loc=[3.0, 3.0], scale=0.2, size=(20, 2))
    X = np.vstack([X1, X2])

    scores = ds.compute_calinski_harabasz_scores(X, cluster_range=[2, 3])

    assert set(scores.keys()) == {2, 3}

    # Compare with sklearn's CH for k=2
    kmeans_2 = KMeans(n_clusters=2, random_state=42, n_init="auto")
    labels_2 = kmeans_2.fit_predict(X)
    ch_2 = calinski_harabasz_score(X, labels_2)

    assert scores[2] == pytest.approx(ch_2, rel=1e-3)


def test_run_kmeans_shapes_and_labels():
    """run_kmeans should return labels and centers with correct shapes."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [5.0, 5.0],
            [5.1, 5.2],
        ]
    )
    labels, centers = ds.run_kmeans(X, n_clusters=2, random_state=0)

    assert isinstance(labels, np.ndarray)
    assert isinstance(centers, np.ndarray)
    assert labels.shape == (X.shape[0],)
    assert centers.shape == (2, X.shape[1])

    # Should produce exactly 2 unique cluster labels
    assert len(np.unique(labels)) == 2


def test_run_hierarchical_clustering():
    """run_hierarchical_clustering should produce n_clusters labels."""
    X = np.array(
        [
            [0.0, 0.0],
            [0.2, -0.1],
            [5.0, 5.0],
            [5.2, 4.9],
            [10.0, 10.0],
        ]
    )
    labels = ds.run_hierarchical_clustering(X, n_clusters=3, linkage="ward")

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X.shape[0],)
    assert len(np.unique(labels)) == 3


# ============================================================
# Part 3: GMM & Outliers
# ============================================================

def test_fit_gmm_and_gmm_cluster_labels():
    """fit_gmm should return a usable GMM model; gmm_cluster_labels should use it."""
    rng = np.random.default_rng(1)
    # Two clear clusters in 1D
    X1 = rng.normal(loc=-2.0, scale=0.3, size=(20, 1))
    X2 = rng.normal(loc=2.0, scale=0.3, size=(20, 1))
    X = np.vstack([X1, X2])

    model = ds.fit_gmm(X, n_components=2, random_state=0)
    labels = ds.gmm_cluster_labels(model, X)

    assert hasattr(model, "predict")
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X.shape[0],)
    assert len(np.unique(labels)) == 2


def test_gmm_outlier_mask():
    """gmm_outlier_mask should detect low-probability points as outliers."""
    rng = np.random.default_rng(2)
    X1 = rng.normal(loc=[0.0, 0.0], scale=1.2, size=(30, 2))
    X2 = rng.normal(loc=[3.0, 3.0], scale=1.2, size=(30, 2))
    X_inliers = np.vstack([X1, X2])

    # Some extreme outliers far away

    X_outliers = np.array(
        [
            [1.0, 2.0],
            [-1, 4],
        ]
    )

    X_all = np.vstack([X_inliers, X_outliers])

    # Fit GMM on all points (or just inliers; either is fine as long as behavior is consistent)
    model = ds.fit_gmm(X_all, n_components=2, random_state=0)

    mask = ds.gmm_outlier_mask(model, X_all, threshold=0.9)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (X_all.shape[0],)

    # Expect at least the extreme points to be flagged as outliers
    assert not np.all(mask), "Some of the samples should not be marked outliers."
    assert mask[-1] or mask[-2], "At least one of the outliers should be flaged"


# ============================================================
# Part 4: Cluster Validation
# ============================================================

def test_silhouette_for_labels_matches_sklearn():
    """silhouette_for_labels should match sklearn's silhouette_score."""
    rng = np.random.default_rng(3)
    X1 = rng.normal(loc=[0.0, 0.0], scale=0.1, size=(10, 2))
    X2 = rng.normal(loc=[2.0, 2.0], scale=0.1, size=(10, 2))
    X = np.vstack([X1, X2])
    labels = np.array([0] * 10 + [1] * 10)

    score_ds = ds.silhouette_for_labels(X, labels)
    score_sk = silhouette_score(X, labels)

    assert score_ds == pytest.approx(score_sk, rel=1e-6)
    assert -1.0 <= score_ds <= 1.0


def test_silhouette_for_labels_single_cluster_returns_nan():
    """
    silhouette_for_labels should return np.nan when fewer than 2 clusters
    are present in the labels.
    """
    X = np.array([[0.0], [1.0], [2.0]])
    labels = np.array([0, 0, 0])

    score = ds.silhouette_for_labels(X, labels)
    assert np.isnan(score)
