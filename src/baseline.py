"""
Classical TF-IDF + Logistic Regression baseline for the sentiment task.

Ports notebook cells 28-30. Reuses `evaluate_sentiment` from
src/sentiment.py so the baseline and the pretrained Transformer are
scored with identical code on the same held-out rows.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from . import config
from .sentiment import evaluate_sentiment


def make_gold_split(
    gold_df: pd.DataFrame,
    test_size: float = config.BASELINE_TEST_SIZE,
    seed: int = config.BASELINE_SPLIT_SEED,
    label_col: str = "gold_label",
):
    """Stratified train/test split of the hand-labeled gold set, used only
    for fitting/evaluating the TF-IDF baseline (the Transformer itself was
    never trained, so it doesn't need a split). Mirrors notebook cell 28.
    """
    train_df, test_df = train_test_split(
        gold_df,
        test_size=test_size,
        random_state=seed,
        stratify=gold_df[label_col],
    )
    return train_df, test_df


def fit_tfidf_baseline(train_texts, train_labels) -> dict:
    """Fit TF-IDF + Logistic Regression on the training split only --
    never on the full gold set before splitting, so test-set vocabulary
    never leaks into the vectorizer. Mirrors notebook cell 28.

    Returns the fitted vectorizer and model together so a caller can't
    accidentally call predict_tfidf_baseline() with a mismatched pair.
    """
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(train_texts)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, train_labels)

    return {"vectorizer": vectorizer, "model": model}


def predict_tfidf_baseline(fitted_objects: dict, texts):
    """Predict with the fitted baseline. Mirrors notebook cell 29."""
    X = fitted_objects["vectorizer"].transform(texts)
    return fitted_objects["model"].predict(X)


def compare_baseline_vs_transformer(test_df: pd.DataFrame, baseline_predictions) -> dict:
    """Evaluate both models on the exact same held-out rows, using the
    shared evaluate_sentiment() helper for both. Mirrors notebook cell 29.
    """
    return {
        "baseline": evaluate_sentiment(test_df["gold_label"], baseline_predictions),
        "transformer": evaluate_sentiment(test_df["gold_label"], test_df["sentiment"]),
    }


def inspect_disagreements(test_df: pd.DataFrame, baseline_predictions) -> pd.DataFrame:
    """Rows where the TF-IDF baseline and the Transformer disagree --
    often more educational than another decimal place of F1. Mirrors
    notebook cell 30.
    """
    test_df_copy = test_df.copy()
    test_df_copy["baseline_pred"] = baseline_predictions
    return test_df_copy[test_df_copy["baseline_pred"] != test_df_copy["sentiment"]]
