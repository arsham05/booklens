"""
Join sentiment + topic outputs into one book-level analysis table and
answer the guide's descriptive cross-analysis questions (Task 9).

Ports notebook cells 41 and 52-56.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from . import config


def get_topical_outliers(
    books_df: pd.DataFrame,
    topic_col: str = "Topic",
    n_samples: Optional[int] = None,
) -> pd.DataFrame:
    """Books that fall into BERTopic's outlier topic (-1). Mirrors
    notebook cell 41. Pass n_samples to get a random sample instead of
    every outlier row.
    """
    outliers = books_df[books_df[topic_col] == -1]
    if n_samples:
        outliers = outliers.sample(
            n=min(n_samples, len(outliers)), random_state=config.INSPECTION_SEED
        )
    return outliers


def aggregate_book_sentiment(sample_df: pd.DataFrame) -> pd.DataFrame:
    """Roll excerpt-level sentiment predictions up to one row per book:
    counts per label plus mean class probabilities. Mirrors notebook
    cell 52.
    """
    return sample_df.groupby("book_id").agg(
        total_excerpts=("excerpt", "count"),
        positive_count=("sentiment", lambda x: (x == "POS").sum()),
        negative_count=("sentiment", lambda x: (x == "NEG").sum()),
        neutral_count=("sentiment", lambda x: (x == "NEU").sum()),
        mean_pos_score=("pos_score", "mean"),
        mean_neg_score=("neg_score", "mean"),
        mean_neu_score=("neu_score", "mean"),
    ).reset_index()


def join_topics_and_sentiment(
    books_df: pd.DataFrame,
    book_sentiment: pd.DataFrame,
    topic_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Merge book metadata + sentiment aggregates + topic human labels
    into one final book_analysis dataframe. Mirrors notebook cell 53.
    """
    book_analysis = pd.merge(books_df, book_sentiment, on="book_id", how="inner")
    book_analysis = pd.merge(
        book_analysis, topic_summary[["Topic", "Human_Label"]], on="Topic", how="left"
    )

    final_columns = [
        "book_id", "title", "author", "genre_clean", "summary_length",
        "Topic", "Human_Label", "total_excerpts",
        "positive_count", "neutral_count", "negative_count",
        "mean_pos_score", "mean_neu_score", "mean_neg_score",
    ]
    return book_analysis[[c for c in final_columns if c in book_analysis.columns]].reset_index(drop=True)


def create_summary_tables(book_analysis: pd.DataFrame) -> dict:
    """Answer the guide's cross-analysis questions (Task 9, item 5) in one
    call: which topics skew most positive, which have the longest/shortest
    summaries, and which books are topical outliers. Mirrors notebook
    cells 54-56.
    """
    topic_pos_sentiment = (
        book_analysis.groupby(["Topic", "Human_Label"])["mean_pos_score"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    topic_length = (
        book_analysis.groupby(["Topic", "Human_Label"])["summary_length"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    topical_outliers = book_analysis[book_analysis["Topic"] == -1][
        ["book_id", "title", "summary_length", "mean_pos_score"]
    ]
    return {
        "topic_pos_sentiment": topic_pos_sentiment,
        "topic_length": topic_length,
        "topical_outliers": topical_outliers,
    }
