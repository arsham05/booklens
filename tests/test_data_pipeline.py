"""
Unit tests for src/data.py (Task 1: project setup and reproducible data).

These tests don't touch the Kaggle mirror at all -- they build small
in-memory dataframes that match the project's column schema (see
data/README.md) and check that cleaning, validation, and plotting behave
the way the pipeline depends on. Run with:

    pytest tests/test_data_pipeline.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as data_mod


def _toy_df() -> pd.DataFrame:
    """A tiny raw dataframe shaped like the output of data_mod.load_books()
    + data_mod.sample_working_set() -- i.e. what clean_books()/validate_books()
    actually receive in the notebook.
    """
    return pd.DataFrame({
        "title": ["Book A", "Book A", "Book B", "Book C", "Book C"],
        "author": ["Author 1", "Author 1", "Author 2", "Author 3", "Author 3"],
        "genre": [
            "['Fiction']", "['Fiction']", "['History']",
            "['Fiction', 'Romance']", "['Fiction', 'Romance']",
        ],
        "summary": [
            "A   messy   summary\nwith line breaks.",
            "A   messy   summary\nwith line breaks.",
            "A clean history summary.",
            "",  # empty after collapsing -> should be dropped
            "&amp; an HTML-escaped summary.",
        ],
        "excerpt": [
            "Excerpt one.", "Excerpt two.", "Excerpt three.",
            "Excerpt four.", "Excerpt five.",
        ],
        "rating": [5.0, 4.0, 3.0, 2.0, 1.0],
        "book_id": [0, 0, 1, 2, 2],
    })


def test_clean_books_collapses_whitespace_and_unescapes_html():
    df = data_mod.clean_books(_toy_df())
    assert "\n" not in df["summary"].iloc[0]
    assert "  " not in df["summary"].iloc[0]
    assert any(s.startswith("& an HTML-escaped") for s in df["summary"])


def test_clean_books_drops_rows_with_empty_summary():
    df = data_mod.clean_books(_toy_df())
    assert df["summary"].str.len().gt(0).all()
    assert len(df) == 4  # one of the five input rows is dropped


def test_clean_books_adds_single_label_genre_clean():
    df = data_mod.clean_books(_toy_df())
    assert "genre_clean" in df.columns
    multi_genre_row = df[df["title"] == "Book C"]
    assert (multi_genre_row["genre_clean"] == "Fiction").all()


def test_validate_books_reports_expected_keys_and_counts():
    report = data_mod.validate_books(_toy_df())
    expected_keys = {
        "shape", "dtypes", "missing_values",
        "duplicate_titles", "duplicate_summaries", "summary_length_stats",
    }
    assert expected_keys.issubset(report.keys())
    assert report["shape"] == (5, 7)
    # book_id already de-dupes titles 1:1, so duplicate_titles is 0.
    assert report["duplicate_titles"] == 0


def test_make_eda_plots_saves_exactly_four_files(tmp_path):
    df = data_mod.clean_books(_toy_df())
    saved_paths = data_mod.make_eda_plots(df, out_dir=str(tmp_path))

    assert len(saved_paths) == 4
    for path in saved_paths:
        assert Path(path).exists()
        assert Path(path).stat().st_size > 0


def test_parse_genre_falls_back_to_raw_value_on_bad_input():
    assert data_mod._parse_genre("Not-a-list") == "Not-a-list"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
