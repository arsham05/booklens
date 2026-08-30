"""
Sentiment inference, gold-set candidate sampling, and evaluation.

Ports notebook cells 19-27. `evaluate_sentiment` is written once here and
reused by src/baseline.py for the TF-IDF comparison, instead of the
notebook's two separate classification_report/confusion_matrix blocks
(cells 25-26 for the Transformer, cell 29 for the baseline).
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import pipeline

from . import config


def load_sentiment_pipeline(
    model_name: str = config.SENTIMENT_MODEL_NAME,
    device: Optional[int] = None,
):
    """Load the pretrained sentiment pipeline. device=None auto-selects
    GPU if available, else CPU. Mirrors notebook cell 19.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        model=model_name,
        tokenizer=model_name,
        top_k=None,  # return scores for all classes, not only the top one
        device=device,
        truncation=True,
        max_length=config.SENTIMENT_MAX_LENGTH,
    )


def predict_sentiment(
    texts: list[str],
    pipe,
    batch_size: int = config.SENTIMENT_BATCH_SIZE,
) -> pd.DataFrame:
    """Run the pipeline over a list of texts; return one row per text with
    the predicted label and per-class scores (e.g. sentiment, POS_score,
    NEG_score, NEU_score).

    The notebook (cell 21) called `pipe()` once per row through
    `.apply()`, i.e. batch_size=1. This version passes the whole list to
    the pipeline with an explicit `batch_size`, which returns identical
    per-text scores but runs faster on the full excerpt table.
    """
    raw_results = pipe(list(texts), batch_size=batch_size)

    rows = []
    for raw_answer in raw_results:
        best_label = max(raw_answer, key=lambda x: x["score"])["label"]
        row = {"sentiment": best_label}
        for item in raw_answer:
            row[f"{item['label']}_score"] = item["score"]
        rows.append(row)

    return pd.DataFrame(rows)


def sample_gold_candidates(
    df: pd.DataFrame,
    n_per_bucket: int = config.GOLD_N_PER_BUCKET,
    n_ambiguous: int = config.GOLD_N_AMBIGUOUS,
    clear_seed: int = config.GOLD_CLEAR_SEED,
    ambiguous_seed: int = config.GOLD_AMBIGUOUS_SEED,
    shuffle_seed: int = config.GOLD_SHUFFLE_SEED,
    uncertain_threshold: float = config.UNCERTAIN_SCORE_THRESHOLD,
) -> pd.DataFrame:
    """Build the ~80-excerpt gold-set candidate sample for manual
    annotation: a balanced sample across rating buckets, plus extra
    excerpts the model itself is unsure about (no class scored above
    `uncertain_threshold`). Mirrors notebook cell 23.

    Do not change the three seeds independently of re-labeling the gold
    set: gold_set_annotated.csv was hand-labeled against the exact rows
    these seeds produce (see the note in src/config.py).
    """
    def rating_bucket(r):
        if r <= 2:
            return "negative"
        if r == 3:
            return "neutral"
        return "positive"

    df = df.copy()
    df["rating_bucket"] = df["rating"].apply(rating_bucket)

    clear_df = df.groupby("rating_bucket", group_keys=False)[df.columns].apply(
        lambda x: x.sample(n=min(len(x), n_per_bucket), random_state=clear_seed)
    )

    score_cols = ["pos_score", "neg_score", "neu_score"]
    ambiguous_pool = df[df[score_cols].max(axis=1) <= uncertain_threshold]
    ambiguous_df = ambiguous_pool.sample(
        n=min(len(ambiguous_pool), n_ambiguous), random_state=ambiguous_seed
    )

    gold_df = (
        pd.concat([clear_df, ambiguous_df])
        .drop_duplicates(subset=["title", "excerpt"])
        .sample(frac=1, random_state=shuffle_seed)
        .reset_index(drop=True)
    )
    return gold_df


def evaluate_sentiment(y_true, y_pred, labels: Optional[list[str]] = None) -> dict:
    """Macro precision/recall/F1, the full classification report, and the
    confusion matrix for any (y_true, y_pred) pair.

    Shared by the pretrained Transformer (notebook cells 25-26) and the
    TF-IDF baseline (notebook cell 29) so the metric logic exists once
    instead of being duplicated for each model.
    """
    labels = labels or config.SENTIMENT_LABELS
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    return {
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "classification_report": classification_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
    }


def inspect_errors(
    gold_df: pd.DataFrame,
    n_pos_as_neu: int = 7,
    n_neg_as_pos: int = 3,
) -> pd.DataFrame:
    """Pull the two most common error types (from the confusion matrix)
    for manual reading. Mirrors notebook cell 27. Pass different counts if
    a rerun's confusion matrix highlights different error cells.
    """
    pos_predicted_neu = gold_df[
        (gold_df["gold_label"] == "POS") & (gold_df["sentiment"] == "NEU")
    ][:n_pos_as_neu]
    pos_predicted_neg = gold_df[
        (gold_df["gold_label"] == "NEG") & (gold_df["sentiment"] == "POS")
    ][:n_neg_as_pos]
    return pd.concat([pos_predicted_neu, pos_predicted_neg])
