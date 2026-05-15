"""
ML4Health 2026 -- LIME for Text Classification  (Section A of notebook.py)
==========================================================================
Task
----
Train a simple text classifier and explain one prediction with LIME.

Instructions
------------
- Complete every function marked with TODO. Do NOT change function signatures.
- Run `pytest tests/ -v` locally to check your implementations.
- Push to `main` only for final submission; save work-in-progress on a branch.
- Record issues and reflections in `experiences.md`.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


LIME_CLASS_NAMES: list[str] = ["negative", "positive"]
DEFAULT_MOVIE_DATASET = Path(__file__).with_name("movie_sentiment_200.csv")


def build_demo_corpus() -> pd.DataFrame:
    """Return the labelled text dataset for the LIME exercise.

    The preferred source is the local `movie_sentiment_200.csv` file provided
    with the assignment. A tiny in-code fallback is kept so the scaffold still
    has a defined shape if that file is absent.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
        - `text`: movie review text
        - `label`: integer label (`0=negative`, `1=positive`)
    """
    if DEFAULT_MOVIE_DATASET.exists():
        df = pd.read_csv(DEFAULT_MOVIE_DATASET)
        expected_columns = ["text", "label"]
        if list(df.columns) != expected_columns:
            raise ValueError(
                f"{DEFAULT_MOVIE_DATASET.name} must have columns {expected_columns}, "
                f"got {list(df.columns)}"
            )
        return df

    texts = [
        ("this movie was dull and forgettable", 0),
        ("the plot felt messy and frustrating", 0),
        ("i liked the performances and the ending", 1),
        ("this was a touching and creative film", 1),
    ]
    return pd.DataFrame(texts, columns=["text", "label"])


def split_corpus(
    data: pd.DataFrame,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split the text corpus into train and test partitions.

    Parameters
    ----------
    data : DataFrame returned by `build_demo_corpus`.
    test_size : fraction reserved for the test split.
    seed : random seed used for the stratified split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # TODO: use train_test_split with stratify=data["label"] and random_state=seed.
    raise NotImplementedError("Implement split_corpus")


def build_text_pipeline(seed: int = 42) -> Pipeline:
    """Create a TF-IDF + logistic-regression text classification pipeline.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with:
        - `TfidfVectorizer(lowercase=True, stop_words="english")`
        - `LogisticRegression(max_iter=1000, random_state=seed)`
    """
    # TODO: return the pipeline described in the docstring.
    raise NotImplementedError("Implement build_text_pipeline")


def fit_text_pipeline(model: Pipeline, X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """Fit the text pipeline and return it."""
    # TODO: fit `model` on `X_train`, `y_train`, then return the fitted model.
    raise NotImplementedError("Implement fit_text_pipeline")


def evaluate_accuracy(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> float:
    """Compute classification accuracy on the test split.

    Returns
    -------
    float
        Accuracy rounded to 4 decimal places.
    """
    # TODO: call model.predict, compute accuracy_score, round to 4 decimals.
    raise NotImplementedError("Implement evaluate_accuracy")


def explain_text_prediction(
    model: Pipeline,
    text: str,
    class_names: list[str] | None = None,
    num_features: int = 6,
) -> list[tuple[str, float]]:
    """Explain one text prediction with LIME.

    Parameters
    ----------
    model : fitted sklearn text pipeline with `predict_proba`
    text : text instance to explain
    class_names : human-readable class names. Defaults to `LIME_CLASS_NAMES`.
    num_features : number of words/features shown in the explanation

    Returns
    -------
    list of (feature, weight)
        Explanation list for the model's predicted class.
    """
    # TODO:
    # 1. Build a LimeTextExplainer with the provided class names.
    # 2. Explain `text` via `model.predict_proba`.
    # 3. Get the predicted class with model.predict([text])[0].
    # 4. Return explanation.as_list(label=predicted_class).
    raise NotImplementedError("Implement explain_text_prediction")


def top_supporting_words(explanation: list[tuple[str, float]]) -> list[str]:
    """Return words/features with positive weight, ordered by descending weight."""
    # TODO: keep only pairs with weight > 0 and return the feature names sorted
    # by descending weight.
    raise NotImplementedError("Implement top_supporting_words")
