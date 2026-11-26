"""
ds.py

Homework 5: Clustering Techniques on Energy Data

You will implement a clustering workflow using the OWID energy dataset
(`owid-energy.csv`), including:

- Data preprocessing and feature selection
- Standardization
- K-Means clustering with Calinski–Harabasz evaluation
- Hierarchical clustering
- Gaussian Mixture Models (GMM) for clustering and outlier detection
- Cluster validation (e.g., silhouette score)

Complete all functions by replacing the `NotImplementedError` with your code.

- Do NOT rename any functions.
- Do NOT change function signatures.
- You may add small helper functions if needed, but do not change the
  public API defined here.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Part 1: Data Preparation & Feature Selection
# ============================================================

def load_energy_data(path: str) -> pd.DataFrame:
    """
    Load the OWID energy dataset from a CSV file.

    Parameters
    ----------
    path : str
        Path to `owid-energy.csv`.

    Returns
    -------
    pd.DataFrame
        The loaded dataset.

    Notes
    -----
    - Use `pd.read_csv(path)`.
    """
    raise NotImplementedError("Task 1: Implement load_energy_data().")


def initial_summary(df: pd.DataFrame) -> dict[str, object]:
    """
    Compute a basic summary of the dataset for initial exploration.

    Parameters
    ----------
    df : pd.DataFrame
        The energy dataset.

    Returns
    -------
    dict[str, object]
        A dictionary containing:
        - 'n_rows': int, number of rows
        - 'n_cols': int, number of columns
        - 'columns': list[str], all column names
        - 'numeric_columns': list[str], columns with numeric dtype
        - 'categorical_columns': list[str], non-numeric columns

    Notes
    -----
    - This is intended to support Task 1 (Data Preprocessing) in the notebook.
    """
    raise NotImplementedError("Task 1: Implement initial_summary().")


def clean_energy_data(
    df: pd.DataFrame,
    subset_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Clean the energy data by handling missing values and duplicates.

    Suggested behavior
    ------------------
    - Drop duplicate rows.
    - If `subset_cols` is provided, drop rows where any of those
      columns are missing (NaN).
    - Otherwise, you may choose a reasonable default strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Raw energy dataset.
    subset_cols : Iterable[str] or None, default None
        Columns that must be non-missing. If None, you may choose a
        default subset (e.g., key energy features).

    Returns
    -------
    pd.DataFrame
        A cleaned copy of the dataset.

    Notes
    -----
    - Do NOT modify `df` in-place; return a new DataFrame.
    """
    raise NotImplementedError("Task 1: Implement clean_energy_data().")


def filter_year_range(
    df: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Filter the dataset to a specific year range [start_year, end_year].

    Parameters
    ----------
    df : pd.DataFrame
        Energy dataset with a 'year' column.
    start_year : int
        Earliest year to keep (inclusive).
    end_year : int
        Latest year to keep (inclusive).

    Returns
    -------
    pd.DataFrame
        Filtered copy of `df` with years in the specified range.
    """
    raise NotImplementedError("Task 1: Implement filter_year_range().")


def select_clustering_features(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    """
    Select a subset of columns to use as clustering features.

    Typical features for this assignment might include per-capita or share
    variables such as:
        - 'fossil_energy_per_capita'
        - 'renewables_energy_per_capita'
        - 'low_carbon_share_energy'
        - 'per_capita_electricity'
        - etc.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned energy dataset.
    feature_cols : Iterable[str]
        Names of columns to keep as clustering features.

    Returns
    -------
    pd.DataFrame
        DataFrame containing only the requested feature columns.

    Notes
    -----
    - Do NOT modify `df` in-place.
    - You may assume all requested columns exist.
    """
    raise NotImplementedError("Task 2: Implement select_clustering_features().")


def standardize_features(
    df_features: pd.DataFrame,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Standardize numeric features to mean 0 and standard deviation 1.

    For each column:
        z = (x - mean) / std

    Parameters
    ----------
    df_features : pd.DataFrame
        DataFrame of numeric features for clustering.

    Returns
    -------
    df_scaled : pd.DataFrame
        DataFrame with the same columns and index as `df_features`,
        but standardized.
    means : np.ndarray
        1D array of column means, in the same order as df_features.columns.
    stds : np.ndarray
        1D array of column standard deviations.

    Notes
    -----
    - Do NOT modify `df_features` in-place.
    - You may use `sklearn.preprocessing.StandardScaler` or implement
      standardization manually with NumPy / Pandas.
    """
    raise NotImplementedError("Task 2: Implement standardize_features().")


# ============================================================
# Part 2: K-Means & Hierarchical Clustering
# ============================================================

def compute_calinski_harabasz_scores(
    X: pd.DataFrame | np.ndarray,
    cluster_range: Iterable[int],
    random_state: int = 42,
) -> dict[int, float]:
    """
    Compute Calinski–Harabasz scores over a range of K values for K-Means.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Standardized feature matrix for clustering.
    cluster_range : Iterable[int]
        Iterable of candidate numbers of clusters, e.g., range(2, 10).
    random_state : int, default 42
        Random seed for K-Means initialization.

    Returns
    -------
    dict[int, float]
        A dictionary mapping n_clusters -> Calinski–Harabasz score.

    Notes
    -----
    - Use `sklearn.cluster.KMeans` and
      `sklearn.metrics.calinski_harabasz_score`.
    - Higher scores indicate better-defined clusters.
    """
    raise NotImplementedError(
        "Task 2: Implement compute_calinski_harabasz_scores()."
    )


def run_kmeans(
    X: pd.DataFrame | np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run K-Means clustering on the feature matrix.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Standardized feature matrix.
    n_clusters : int
        Number of clusters.
    random_state : int, default 42
        Random seed for K-Means initialization.

    Returns
    -------
    labels : np.ndarray
        1D array of cluster labels for each row in X.
    centers : np.ndarray
        2D array of cluster centers in feature space.

    Notes
    -----
    - Use `sklearn.cluster.KMeans`.
    - Do not perform any plotting here; this function should just
      return numeric results.
    """
    raise NotImplementedError("Task 2: Implement run_kmeans().")


def run_hierarchical_clustering(
    X: pd.DataFrame | np.ndarray,
    n_clusters: int,
    linkage: str = "ward",
) -> np.ndarray:
    """
    Perform hierarchical (agglomerative) clustering.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Standardized feature matrix.
    n_clusters : int
        Desired number of clusters.
    linkage : str, default "ward"
        Linkage criterion (e.g., "ward", "complete", "average").

    Returns
    -------
    labels : np.ndarray
        1D array of cluster labels for each row in X.

    Notes
    -----
    - Use `sklearn.cluster.AgglomerativeClustering`.
    - Dendrogram plotting should be done in the notebook,
      not inside this function.
    """
    raise NotImplementedError("Task 3: Implement run_hierarchical_clustering().")


# ============================================================
# Part 3: Gaussian Mixture Models (GMM) & Outliers
# ============================================================

def fit_gmm(
    X: pd.DataFrame | np.ndarray,
    n_components: int,
    random_state: int = 42,
) -> object:
    """
    Fit a Gaussian Mixture Model (GMM) to the feature matrix.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Standardized feature matrix.
    n_components : int
        Number of mixture components (clusters).
    random_state : int, default 42
        Random seed for initialization.

    Returns
    -------
    model : object
        Fitted GMM model (e.g., sklearn.mixture.GaussianMixture).

    Notes
    -----
    - Do not compute outliers here; that is handled in a separate function.
    """
    raise NotImplementedError("Task 3: Implement fit_gmm().")


def gmm_cluster_labels(
    model: object,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    Predict cluster labels for each data point using a fitted GMM.

    Parameters
    ----------
    model : object
        Fitted GMM model with a `.predict` method.
    X : pd.DataFrame or np.ndarray
        Standardized feature matrix.

    Returns
    -------
    np.ndarray
        1D array of cluster labels.
    """
    raise NotImplementedError("Task 3: Implement gmm_cluster_labels().")


def gmm_outlier_mask(
    model: object,
    X: pd.DataFrame | np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """
    Identify outliers based on GMM cluster membership probabilities.

    A common approach:
    - Use `model.predict_proba(X)` to obtain responsibilities.
    - For each row, compute the maximum responsibility across clusters.
    - Mark points as outliers if max responsibility < `threshold`.

    Parameters
    ----------
    model : object
        Fitted GMM model with a `.predict_proba` method.
    X : pd.DataFrame or np.ndarray
        Standardized feature matrix.
    threshold : float, default 0.01
        Probability threshold below which points are considered outliers.

    Returns
    -------
    np.ndarray
        1D boolean array where True indicates an outlier.

    Notes
    -----
    - This supports Task 3 (Outlier Detection and Analysis).
    """
    raise NotImplementedError("Task 3: Implement gmm_outlier_mask().")


# ============================================================
# Part 4: Cluster Validation
# ============================================================

def silhouette_for_labels(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
) -> float:
    """
    Compute the silhouette score for a given clustering.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Standardized feature matrix.
    labels : np.ndarray
        Cluster labels for each row in X.

    Returns
    -------
    float
        Silhouette score (between -1 and 1).

    Notes
    -----
    - Use `sklearn.metrics.silhouette_score`.
    - If there are fewer than 2 clusters in `labels`, you may choose to
      return `np.nan` or raise a `ValueError`. Be consistent.
    """
    raise NotImplementedError("Task 4: Implement silhouette_for_labels().")
