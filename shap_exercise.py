"""
ML4Health 2026 -- SHAP on Breast Cancer Wisconsin  (Section B of notebook.py)
=============================================================================
Task
----
Train a tabular classifier and inspect both global and local SHAP explanations.

Instructions
------------
- Complete every function marked with TODO. Do NOT change function signatures.
- Run `pytest tests/ -v` locally to check your implementations.
- Push to `main` only for final submission; save work-in-progress on a branch.
- Record issues and reflections in `experiences.md`.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_breast_cancer_dataframe() -> tuple[pd.DataFrame, pd.Series]:
    """Load the scikit-learn Breast Cancer Wisconsin dataset.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix with original feature names.
    y : pandas.Series
        Binary target named `target`.
    """
    # TODO: use `load_breast_cancer(as_frame=True)` and return the feature
    # DataFrame plus the target series.
    raise NotImplementedError("Implement load_breast_cancer_dataframe")


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train/test split."""
    # TODO: use train_test_split with stratify=y and random_state=seed.
    raise NotImplementedError("Implement split_dataset")


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = 42,
) -> RandomForestClassifier:
    """Train a simple random-forest classifier.

    Use `RandomForestClassifier(n_estimators=200, random_state=seed)`.
    """
    # TODO: instantiate the model, fit it, and return the fitted estimator.
    raise NotImplementedError("Implement train_random_forest")


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    """Compute test accuracy rounded to 4 decimal places."""
    # TODO: call model.predict, compute accuracy_score, and round to 4 decimals.
    raise NotImplementedError("Implement evaluate_model")


def make_tree_explainer(model: RandomForestClassifier) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for the fitted random forest."""
    # TODO: return shap.TreeExplainer(model)
    raise NotImplementedError("Implement make_tree_explainer")


def compute_shap_values(
    explainer: shap.TreeExplainer,
    X: pd.DataFrame,
):
    """Compute SHAP values for a DataFrame of samples.

    Returns
    -------
    object
        The result of calling `explainer(X)`.
    """
    # TODO: return explainer(X)
    raise NotImplementedError("Implement compute_shap_values")


def mean_absolute_shap_importance(shap_values, feature_names: list[str]) -> pd.Series:
    """Compute global feature importance from SHAP values.

    For binary classification with modern SHAP APIs, `shap_values.values` is
    typically shaped `(n_samples, n_features, n_classes)`. Use the positive
    class (`class index 1`) if a class dimension is present. Then compute the
    mean absolute SHAP value per feature.

    Returns
    -------
    pandas.Series
        Indexed by feature name, sorted descending.
    """
    # TODO:
    # 1. Extract the raw values array.
    # 2. If it is 3D, keep the positive-class slice `values[:, :, 1]`.
    # 3. Compute mean absolute SHAP value per feature.
    # 4. Return a descending pandas Series indexed by `feature_names`.
    raise NotImplementedError("Implement mean_absolute_shap_importance")


def explain_one_instance(shap_values, row_index: int = 0) -> np.ndarray:
    """Return the SHAP values for one sample.

    If a class dimension is present, return the positive-class explanation.
    """
    # TODO: slice one row from shap_values.values; handle both 2D and 3D arrays.
    raise NotImplementedError("Implement explain_one_instance")
