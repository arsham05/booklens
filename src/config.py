"""
Central configuration for the BookLens pipeline.

Every model name, file path, random seed, batch size, and analysis
threshold that was a bare literal inside a notebook cell now lives here,
named and commented, so a rerun is a one-line edit instead of a search
through 56 cells.

Random seeds are intentionally NOT unified into a single SEED constant.
The notebook used three different seeds while building the gold-set
candidate sample (cell 23: seed 53 for the balanced "clear" sample, seed
49 for the "ambiguous" sample, seed 42 for the final shuffle). That
candidate file was then hand-labeled outside the notebook to produce
gold_set_annotated.csv. If the seeds were merged into one value, a future
run would generate a *different* set of candidate excerpts that no longer
lines up with the labels a person already spent time writing by hand. Each
seed below is kept separate and documented with the step it belongs to.
"""

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Relative to the repository root (see Section 8, Repository Structure).
# The original Colab notebook prefixed these with "booklens/" because it
# had just git-cloned the repo into a subfolder of /content; that prefix
# does not apply once this code lives inside the repo itself.
RAW_DATA_DIR = "data"
BOOKS_DATA_FILENAME = "books_data.csv"        # Kaggle mirror: book metadata
REVIEWS_FILENAME = "Books_rating.csv"         # Kaggle mirror: per-review ratings/text

CLEAN_SAMPLE_PATH = "data/clean_books_sample.csv"
GOLD_CANDIDATE_PATH = "data/gold_set_candidate.csv"   # written for manual labeling
GOLD_ANNOTATED_PATH = "data/gold_set_annotated.csv"   # read back in after labeling

FIGURES_DIR = "outputs/figures"
METRICS_DIR = "outputs/metrics"
TOPICS_DIR = "outputs/topics"

# ---------------------------------------------------------------------------
# Task 1 — sampling the working set (notebook cell 7)
# ---------------------------------------------------------------------------
N_BOOKS = 220
EXCERPTS_PER_BOOK = 5
SAMPLE_SEED = 42

# ---------------------------------------------------------------------------
# Task 1 — EDA (notebook cell 14)
# ---------------------------------------------------------------------------
TOP_N_GENRES = 6  # top genres shown individually in the pie chart; rest -> "Other"

# ---------------------------------------------------------------------------
# Task 2 — sentiment inference (notebook cells 18-21)
# ---------------------------------------------------------------------------
SENTIMENT_MODEL_NAME = "finiteautomata/bertweet-base-sentiment-analysis"
SENTIMENT_MAX_LENGTH = 128
# Not present in the original notebook, which called the pipeline once per
# row via .apply() (effectively batch_size=1). Batching does not change any
# individual excerpt's predicted scores -- it only changes how many texts
# are pushed through the model at once -- so this is a safe speed-up, not a
# behavior change. Lower this if you hit GPU/CPU memory limits.
SENTIMENT_BATCH_SIZE = 32
SENTIMENT_LABELS = ["NEG", "NEU", "POS"]
SENTIMENT_DISPLAY_LABELS = ["Negative", "Neutral", "Positive"]

# ---------------------------------------------------------------------------
# Task 3 — gold-set candidate sampling (notebook cell 23)
# ---------------------------------------------------------------------------
GOLD_N_PER_BUCKET = 20      # per rating_bucket (negative/neutral/positive)
GOLD_N_AMBIGUOUS = 20       # extra excerpts the model itself is unsure about
GOLD_CLEAR_SEED = 53
GOLD_AMBIGUOUS_SEED = 49
GOLD_SHUFFLE_SEED = 42
UNCERTAIN_SCORE_THRESHOLD = 0.7  # max class score <= this -> "ambiguous"

# ---------------------------------------------------------------------------
# Task 4 — TF-IDF baseline (notebook cell 28)
# ---------------------------------------------------------------------------
BASELINE_TEST_SIZE = 0.2
BASELINE_SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Task 5 — preparing summaries for topic discovery (notebook cell 31)
# ---------------------------------------------------------------------------
MIN_SUMMARY_LENGTH = 50  # characters; shorter summaries are dropped

# ---------------------------------------------------------------------------
# Task 6 — embeddings -> UMAP -> HDBSCAN (notebook cells 32-34)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
# sentence-transformers' own default is already 32; set explicitly here so
# it is a config knob instead of a silent library default.
EMBEDDING_BATCH_SIZE = 32

UMAP_SEED = 42
UMAP_N_COMPONENTS = 5       # for clustering
UMAP_N_COMPONENTS_2D = 2    # separate projection, for the visualization only
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"

HDBSCAN_METRIC = "euclidean"
HDBSCAN_SELECTION_METHOD = "eom"
MIN_CLUSTER_SIZE = 5

# ---------------------------------------------------------------------------
# Task 8 — one-parameter sensitivity check (notebook cell 50)
# ---------------------------------------------------------------------------
SENSITIVITY_MIN_CLUSTER_SIZE = 10  # only this changes; everything else stays fixed

# ---------------------------------------------------------------------------
# Shared inspection seed
# ---------------------------------------------------------------------------
# Every manual-inspection sample in the notebook (cluster inspection cell 36,
# outlier inspection cell 41, topic inspection cell 48) used random_state=42.
INSPECTION_SEED = 42
