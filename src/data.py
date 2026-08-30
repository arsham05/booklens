"""
Data acquisition, cleaning, validation, and EDA plotting.

Ports notebook cells 5-17 into reusable functions. The Colab-only setup
cells (mounting GH_TOKEN/KAGGLE secrets, `!git clone`, `!kaggle datasets
download`, `!unzip`) are deliberately NOT ported here: they are one-off
environment/authentication steps, not pipeline logic, and stay documented
as manual setup instructions (see data/README.md and Task 1 of the guide)
instead of being wrapped in a function that still needs a live Colab
runtime to work.
"""

from __future__ import annotations

import ast
import html
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config


def load_books(data_dir: str = config.RAW_DATA_DIR) -> pd.DataFrame:
    """Load the two raw Kaggle CSV mirror tables and merge them into one
    excerpt-level dataframe using the project's column schema.

    Mirrors notebook cells 5-6: read books_data.csv + Books_rating.csv,
    inner-join on Title, rename to the project schema, and add a stable
    numeric book_id via pd.factorize on title.
    """
    data_dir = Path(data_dir)
    books_details = pd.read_csv(data_dir / config.BOOKS_DATA_FILENAME)
    reviews = pd.read_csv(data_dir / config.REVIEWS_FILENAME)

    merged_df = pd.merge(books_details, reviews, on="Title", how="inner")
    merged_df = merged_df.rename(columns={
        "Title": "title",
        "authors": "author",
        "categories": "genre",
        "description": "summary",
        "review/text": "excerpt",
        "review/score": "rating",
    })

    final_columns = ["title", "author", "genre", "summary", "excerpt", "rating"]
    df = merged_df[final_columns].copy()
    df["book_id"] = pd.factorize(df["title"])[0]
    return df


def sample_working_set(
    df: pd.DataFrame,
    n_books: int = config.N_BOOKS,
    excerpts_per_book: int = config.EXCERPTS_PER_BOOK,
    seed: int = config.SAMPLE_SEED,
) -> pd.DataFrame:
    """Draw a manageable working sample: up to n_books books, up to
    excerpts_per_book excerpts each. Mirrors notebook cell 7.
    """
    df = df.dropna(subset=["summary", "excerpt", "rating"])

    unique_books = df.drop_duplicates(subset="book_id")
    selected_ids = unique_books.sample(
        n=min(n_books, len(unique_books)), random_state=seed
    )["book_id"]

    pool_df = df[df["book_id"].isin(selected_ids)]
    # [pool_df.columns] avoids a pandas FutureWarning about operating on
    # the grouping column.
    sample_df = (
        pool_df.groupby("book_id", group_keys=False)[pool_df.columns]
        .apply(lambda x: x.sample(n=min(len(x), excerpts_per_book), random_state=seed))
        .reset_index(drop=True)
    )
    return sample_df


def _clean_text(text):
    if not isinstance(text, str):
        return text
    return " ".join(text.split())  # collapse repeated whitespace/newlines


def _parse_genre(g):
    """Kaggle's `categories` column is a stringified Python list, e.g.
    "['Fiction']" -> keep just the first genre for a simple single-label view."""
    try:
        parsed = ast.literal_eval(g)
        if isinstance(parsed, list) and parsed:
            return parsed[0]
    except (ValueError, SyntaxError):
        pass
    return g


def clean_books(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative cleaning: collapse whitespace, unescape HTML entities,
    drop now-empty rows, and add a single-label `genre_clean` column.

    Mirrors notebook cells 12 and 14. Genre parsing is folded in here
    (rather than left inline in the EDA cell) so every downstream
    consumer -- EDA, topic prep, final analysis -- sees the same
    `genre_clean` column instead of three separate re-derivations of it.
    """
    df = df.copy()
    df["summary"] = df["summary"].apply(_clean_text).apply(html.unescape)
    df["excerpt"] = df["excerpt"].apply(_clean_text).apply(html.unescape)

    # Treat now-empty strings as missing and drop them.
    df.replace("", np.nan, inplace=True)
    df = df.dropna(subset=["summary", "excerpt"]).reset_index(drop=True)

    df["genre_clean"] = df["genre"].apply(_parse_genre)
    return df


def validate_books(df: pd.DataFrame) -> dict:
    """Shape, dtypes, missing values, duplicate titles/summaries
    (book-level), and summary text-length stats. Mirrors notebook cells
    8-11. Call this BEFORE clean_books(), same as the notebook order, so
    it reports on the raw sampled data.
    """
    unique_books = df.drop_duplicates(subset="book_id")

    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "duplicate_titles": int(unique_books["title"].duplicated().sum()),
        "duplicate_summaries": int(unique_books["summary"].duplicated().sum()),
        "summary_length_stats": unique_books["summary"].apply(len).describe().to_dict(),
    }


def make_eda_plots(df: pd.DataFrame, out_dir: str = config.FIGURES_DIR) -> list[str]:
    """Save exactly 4 EDA plots and return their file paths:
      1. book summary length histogram
      2. genre distribution pie chart (top N genres + Other)
      3. unique vs duplicate summaries bar chart (book-level)
      4. excerpt length histogram

    Mirrors notebook cells 13, 14, 15, 16. Call this AFTER clean_books()
    -- it needs the `genre_clean` column clean_books() adds.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []

    unique_books = df.drop_duplicates(subset="book_id")

    # 1. Summary length distribution (book-level, not excerpt-level, so
    # each book is only counted once).
    summary_lengths = unique_books["summary"].apply(len)
    fig, ax = plt.subplots()
    summary_lengths.plot.hist(bins=12, alpha=0.5, ax=ax)
    ax.set_title("Distribution of Book Summary Lengths")
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Frequency")
    path = out_path / "01_summary_length_hist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(path))

    # 2. Genre distribution pie chart (one row per book, so genres aren't
    # over-counted by how many excerpts were sampled per book).
    counts = unique_books["genre_clean"].value_counts()
    if len(counts) > config.TOP_N_GENRES:
        top = counts.iloc[: config.TOP_N_GENRES]
        other = pd.Series({"Other": counts.iloc[config.TOP_N_GENRES :].sum()})
        counts = pd.concat([top, other])
    fig, ax = plt.subplots()
    counts.plot.pie(autopct="%1.1f%%", ylabel="", ax=ax)
    path = out_path / "02_genre_distribution_pie.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(path))

    # 3. Unique vs duplicate summaries (book-level). Duplicate *titles*
    # would always be 0 here since book_id is built by factorizing title;
    # duplicate *summaries* is the metric that can actually vary.
    duplicate_summaries = unique_books["summary"].duplicated().sum()
    unique_summaries_count = unique_books["summary"].nunique()
    plot_data = pd.Series(
        [unique_summaries_count, duplicate_summaries],
        index=["Unique Summaries", "Duplicate Summaries"],
    )
    fig, ax = plt.subplots()
    plot_data.plot.bar(rot=0, color=["skyblue", "lightcoral"], ax=ax)
    ax.set_title("Comparison of Unique and Duplicate Summaries (Book Level)")
    ax.set_ylabel("Count")
    path = out_path / "03_unique_vs_duplicate_summaries.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(path))

    # 4. Excerpt (review) length distribution.
    excerpt_lengths = df["excerpt"].apply(len)
    fig, ax = plt.subplots()
    excerpt_lengths.plot.hist(bins=15, color="mediumseagreen", alpha=0.7, ax=ax)
    ax.set_title("Distribution of Excerpt (Review) Lengths")
    ax.set_xlabel("Character Count")
    ax.set_ylabel("Frequency")
    path = out_path / "04_excerpt_length_hist.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(str(path))

    return saved_paths
