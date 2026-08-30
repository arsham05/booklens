"""
Topic discovery: embeddings -> UMAP -> HDBSCAN, then BERTopic on top.

Ports notebook cells 31-50: preparing summaries (Task 5), the explicit
three-step clustering pipeline (Task 6), the BERTopic wrapper and topic
labeling (Task 7), cluster/topic inspection, and the one-parameter
sensitivity check (Task 8). The 2D-visualization cell (46) and the
crosstab comparison (44) are left as short notebook/analysis-script glue
since they're one-off display code rather than reusable pipeline steps.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional

import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

from . import config


def prepare_documents(
    sample_df: pd.DataFrame,
    min_summary_length: int = config.MIN_SUMMARY_LENGTH,
) -> pd.DataFrame:
    """One row per book (deduplicated by summary), filtered to summaries
    at least `min_summary_length` characters. Mirrors notebook cell 31.
    """
    books_df = (
        sample_df[["book_id", "title", "author", "genre_clean", "summary"]]
        .drop_duplicates(subset=["book_id"])
        .dropna(subset=["summary"])
        .copy()
    )
    books_df = books_df.drop_duplicates(subset="summary").reset_index(drop=True)
    books_df["summary_length"] = books_df["summary"].str.strip().str.len()
    books_df = books_df[books_df["summary_length"] >= min_summary_length].reset_index(drop=True)
    return books_df


def embed_documents(
    documents: list[str],
    model_name: str = config.EMBEDDING_MODEL_NAME,
    batch_size: int = config.EMBEDDING_BATCH_SIZE,
):
    """Encode book summaries with a sentence-transformer model. Mirrors
    notebook cell 32; batch_size is now an explicit config value instead
    of an implicit library default.
    """
    embedding_model = SentenceTransformer(model_name)
    embeddings = embedding_model.encode(
        documents, batch_size=batch_size, show_progress_bar=True
    )
    return embedding_model, embeddings


def cluster_documents(
    embeddings,
    seed: int = config.UMAP_SEED,
    min_cluster_size: int = config.MIN_CLUSTER_SIZE,
    n_components: int = config.UMAP_N_COMPONENTS,
):
    """Explicit three-step pipeline: UMAP dimensionality reduction (for
    clustering, not visualization) then HDBSCAN density clustering.
    Mirrors notebook cells 33-34.

    Returns the fitted umap_model and hdbscan_model too -- both are reused
    as-is when building BERTopic in fit_bertopic(), so the manual pipeline
    and BERTopic stay aligned.
    """
    umap_model = UMAP(
        n_components=n_components,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        random_state=seed,
    )
    reduced_embeddings = umap_model.fit_transform(embeddings)

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=config.HDBSCAN_METRIC,
        cluster_selection_method=config.HDBSCAN_SELECTION_METHOD,
    )
    cluster_labels = hdbscan_model.fit_predict(reduced_embeddings)
    return umap_model, hdbscan_model, cluster_labels


def cluster_stats(cluster_labels) -> dict:
    """Cluster count, outlier count, largest/smallest cluster size.
    Mirrors notebook cell 35.
    """
    cluster_labels = list(cluster_labels)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_outliers = cluster_labels.count(-1)
    valid_clusters = [label for label in cluster_labels if label != -1]
    frequencies = Counter(valid_clusters)

    if not frequencies:
        return {
            "n_clusters": 0,
            "n_outliers": n_outliers,
            "largest_cluster_size": 0,
            "smallest_cluster_size": 0,
        }

    largest_cluster_size = frequencies.most_common(1)[0][1]
    smallest_cluster_id = min(frequencies, key=frequencies.get)
    return {
        "n_clusters": n_clusters,
        "n_outliers": n_outliers,
        "largest_cluster_size": largest_cluster_size,
        "smallest_cluster_size": frequencies[smallest_cluster_id],
    }


def inspect_clusters(
    books_df: pd.DataFrame,
    cluster_col: str = "cluster",
    n_examples: int = 3,
    top_n_clusters: int = 3,
    seed: int = config.INSPECTION_SEED,
) -> pd.DataFrame:
    """Sample a few titles + summaries from the largest clusters plus the
    outlier group, for manual reading -- clusters should never be judged
    "good" from the 2D plot alone. Mirrors notebook cell 36.

    Returns a single (cluster_id, title, summary) dataframe instead of
    printing, so it's usable outside a notebook (e.g. saved to
    outputs/topics/).
    """
    cluster_sizes = Counter(c for c in books_df[cluster_col] if c != -1)
    top_clusters = [c for c, _ in cluster_sizes.most_common(top_n_clusters)]
    clusters_to_inspect = top_clusters + [-1]

    rows = []
    for cluster_id in clusters_to_inspect:
        cluster_books = books_df[books_df[cluster_col] == cluster_id]
        sample_books = cluster_books.sample(
            n=min(n_examples, len(cluster_books)), random_state=seed
        )
        for _, row in sample_books.iterrows():
            rows.append({
                "cluster_id": cluster_id,
                "title": row["title"],
                "summary": row["summary"],
            })
    return pd.DataFrame(rows)


def fit_bertopic(documents: list[str], embeddings, embedding_model, umap_model, hdbscan_model):
    """Wrap the same embedding/UMAP/HDBSCAN objects from the manual
    pipeline into BERTopic, so its cluster/topic assignments align with
    cluster_documents() above. Mirrors notebook cell 38.
    """
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=True,
    )
    topics, probs = topic_model.fit_transform(documents, embeddings)
    return topic_model, topics, probs


def build_topic_summary(topic_model, books_df: pd.DataFrame, human_labels: dict) -> pd.DataFrame:
    """Join BERTopic's topic_info table with hand-written human labels and
    up to two representative titles per topic. Mirrors notebook cells
    39-40.

    `human_labels` (e.g. {0: "War, Espionage & Military History", -1:
    "Outliers / Uncategorized"}) is written by hand after reading each
    topic's representative documents -- pass in the mapping for this run
    rather than hardcoding it here, since topic IDs are arbitrary and can
    shift between reruns.
    """
    summary_to_title = dict(zip(books_df["summary"], books_df["title"]))

    def get_top_two_titles(docs_list):
        if isinstance(docs_list, list):
            titles = [summary_to_title.get(doc, "Unknown Title") for doc in docs_list[:2]]
            return " | ".join(titles)
        return ""

    topic_info = topic_model.get_topic_info()
    topic_info["Representative_Titles"] = topic_info["Representative_Docs"].apply(get_top_two_titles)

    analysis_table = pd.DataFrame(list(human_labels.items()), columns=["Topic", "Human_Label"])
    return pd.merge(topic_info, analysis_table, on="Topic")


def sensitivity_check(
    documents: list[str],
    embeddings,
    embedding_model,
    umap_model,
    baseline_topics: list[int],
    min_cluster_size: int = config.SENSITIVITY_MIN_CLUSTER_SIZE,
) -> dict:
    """Rerun BERTopic with only min_cluster_size changed (a single
    parameter, not a grid sweep) and report before/after cluster count and
    outlier ratio. Mirrors notebook cell 50.
    """
    hdbscan_model_new = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=config.HDBSCAN_METRIC,
        cluster_selection_method=config.HDBSCAN_SELECTION_METHOD,
    )
    topic_model_new = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model_new,
        verbose=True,
    )
    topics_new, _ = topic_model_new.fit_transform(documents, embeddings)

    baseline_outliers = baseline_topics.count(-1)
    baseline_clusters = len(set(baseline_topics)) - (1 if -1 in baseline_topics else 0)
    new_outliers = topics_new.count(-1)
    new_clusters = len(set(topics_new)) - (1 if -1 in topics_new else 0)
    total_docs = len(topics_new)

    return {
        "baseline_clusters": baseline_clusters,
        "baseline_outliers": baseline_outliers,
        "baseline_outlier_ratio": baseline_outliers / len(baseline_topics),
        "new_clusters": new_clusters,
        "new_outliers": new_outliers,
        "new_outlier_ratio": new_outliers / total_docs,
        "topic_model": topic_model_new,
        "topics": topics_new,
    }
