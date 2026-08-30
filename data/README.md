# `data/` — Provenance & Schema

This folder is git-ignored by default (see `.gitignore`) except for the small,
redistributable files listed below. The full working dataset is never
committed — it's downloaded fresh from Kaggle by anyone who reruns the
notebook.

## Source

| | |
|---|---|
| **Provider** | Kaggle CSV mirror — [`mohamedbakhet/amazon-books-reviews`](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) |
| **Underlying data** | Amazon book metadata + user review text (the same underlying source the original BookLens guide reaches via raw `.json.gz` files — this project uses the pre-cleaned Kaggle mirror instead, per the Lean scope) |
| **Files** | `books_data.csv` (book metadata) + `Books_rating.csv` (per-review ratings/text) |
| **License / usage** | Redistributed on Kaggle under its listed terms; this repo does **not** re-upload the raw files, only a 500-row non-review-score preview (`sample_books.csv`) and the 80-row hand-labeled gold set, both far short of a meaningful re-publication of the source |

Download it yourself with:

```bash
kaggle datasets download -d mohamedbakhet/amazon-books-reviews --force
unzip -o amazon-books-reviews.zip -d data/
```

This requires a Kaggle API token (`~/.kaggle/kaggle.json` or the
`KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables).

## Raw → project schema

`src/data.py:load_books()` inner-joins the two raw files on `Title` and
renames columns to the project's working schema:

| Raw column (Kaggle) | Project column | Notes |
|---|---|---|
| `Title` | `title` | join key |
| `authors` | `author` | |
| `categories` | `genre` | stringified Python list, e.g. `"['Fiction']"` — parsed down to one label in `genre_clean` by `clean_books()` |
| `description` | `summary` | one per book — feeds the topic-modeling branch |
| `review/text` | `excerpt` | one row per review — feeds the sentiment branch |
| `review/score` | `rating` | 1–5, used only to build a balanced gold-set candidate sample (Task 3) |
| *(derived)* | `book_id` | `pd.factorize(title)`, added after the join |

## Working sample

`src/data.py:sample_working_set()` draws a manageable slice for the whole
project (`src/config.py`):

- `N_BOOKS = 220` books, sampled with `SAMPLE_SEED = 42`
- up to `EXCERPTS_PER_BOOK = 5` reviews per book

On the recorded run this produced **748 excerpt rows** across the 220 books
(some books have fewer than 5 eligible reviews after dropping missing
`summary` / `excerpt` / `rating`).

## Files in this folder

| File | Committed? | What it is |
|---|---|---|
| `sample_books.csv` | ✅ yes | Tiny **redistributable preview only** — 500 rows, `title/author/genre/summary/excerpt`, no ratings. Used to demo the cleaning + EDA pipeline (see `outputs/figures/`) without needing the Kaggle download. **Not** the 748-excerpt working set used for the recorded sentiment/topic results. |
| `gold_set_annotated.csv` | ✅ yes | The hand-labeled ~80-excerpt gold set (Task 3): `title, author, genre, summary, excerpt, rating, book_id, sentiment, pos_score, neg_score, neu_score, rating_bucket, gold_label`. `gold_label` is the human annotation; `sentiment` + the `*_score` columns are the pretrained model's predictions, kept side by side but never edited against each other. |
| `books_data.csv`, `Books_rating.csv` | ❌ git-ignored | Raw Kaggle mirror files — download them yourself (see above). |
| `clean_books_sample.csv` | ❌ git-ignored | Cleaned 748-row working set, written by the notebook (`config.CLEAN_SAMPLE_PATH`) — regenerable from the raw files. |
| `gold_set_candidate.csv` | ❌ git-ignored | The *unlabeled* ~80-row candidate sample (`sentiment_mod.sample_gold_candidates()`), written before manual annotation. Regenerable, since the three sampling seeds (`GOLD_CLEAR_SEED=53`, `GOLD_AMBIGUOUS_SEED=49`, `GOLD_SHUFFLE_SEED=42`) are fixed in `src/config.py`. **Do not change those seeds independently of re-labeling `gold_set_annotated.csv`** — the existing labels were written against the exact rows those seeds produce. |

No full book text, copyrighted excerpts beyond short user-review snippets, or
model weights are committed anywhere in this repo.
