from datasets import load_dataset
import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

# ----------------------------
# Utility functions
# ----------------------------
def normalize_text(text: str) -> str:
    """
    Robust text normalization for scientific/medical text.

    Steps:
    1. Lowercase text
    2. Remove URLs
    3. Remove DOI, citations, or reference patterns like [1], (Smith et al., 2020)
    4. Remove special characters and punctuation (keep alphanumeric + space)
    5. Normalize whitespace
    6. Strip leading/trailing spaces
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove DOI patterns
    text = re.sub(r'doi:\s*\S+', '', text)

    # Remove citations like [1], [12-15], (Smith et al., 2020)
    text = re.sub(r'\[\d+(?:[-,]\d+)*\]', '', text)
    text = re.sub(r'\([a-zA-Z\s]+ et al\.,?\s*\d{4}\)', '', text)

    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Normalize multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ----------------------------
# Load official MTEB dataset
# ----------------------------
print("Loading mteb/trec-covid dataset...")
ds_default = load_dataset("mteb/trec-covid", "default")
ds_corpus = load_dataset("mteb/trec-covid", "corpus")
ds_queries = load_dataset("mteb/trec-covid", "queries")

# Convert to DataFrames
corpus_df = pd.DataFrame(ds_corpus["corpus"])
queries_df = pd.DataFrame(ds_queries["queries"])
qrels_df = pd.DataFrame(ds_default["test"])

# ----------------------------
# Rename columns for consistency
# ----------------------------
corpus_df.rename(columns={"_id": "corpus_id"}, inplace=True)
queries_df.rename(columns={"_id": "query_id"}, inplace=True)
qrels_df.rename(columns={"query-id": "query_id", "corpus-id": "corpus_id", "score": "rel"}, inplace=True)

print("\nDataset Overview:")
print(f"Corpus: {len(corpus_df)} records | Columns: {list(corpus_df.columns)}")
print(f"Queries: {len(queries_df)} records | Columns: {list(queries_df.columns)}")
print(f"Qrels: {len(qrels_df)} records | Columns: {list(qrels_df.columns)}")

# ----------------------------
# Normalize and clean text
# ----------------------------
print("\nCleaning, normalizing, and filtering corpus text...")

# Normalize
corpus_df["text"] = corpus_df["text"].astype(str).map(normalize_text)
queries_df["text"] = queries_df["text"].astype(str).map(normalize_text)
#
# # Remove duplicates and empty strings
corpus_df = corpus_df[corpus_df["text"].str.strip() != ""]
queries_df = queries_df[queries_df["text"].str.strip() != ""]
# corpus_df = corpus_df.drop_duplicates(subset=["text"]).dropna(subset=["text"])
# queries_df = queries_df.drop_duplicates(subset=["text"]).dropna(subset=["text"])


# Filter corpus to max 300 tokens
corpus_df = corpus_df[corpus_df["text"].str.split().str.len() <= 300]

print(f"Cleaned and filtered Corpus (≤300 tokens): {len(corpus_df)}")
print(f"Cleaned Queries: {len(queries_df)}")


# ----------------------------
# Create full dataset folder (before splitting)
# ----------------------------
print("\nCreating full dataset folder...")
os.makedirs("full", exist_ok=True)

# Filter qrels for full dataset
full_qrels = qrels_df[
    qrels_df["query_id"].isin(queries_df["query_id"]) &
    qrels_df["corpus_id"].isin(corpus_df["corpus_id"])
]
original_full_qrels_count = len(full_qrels)

# Save full corpus
corpus_file = os.path.join("full", "corpus.tsv")
corpus_df[["corpus_id", "text"]].to_csv(corpus_file, sep="\t", index=False)

# Save full queries
queries_file = os.path.join("full", "queries.tsv")
queries_df[["query_id", "text"]].rename(columns={"text": "query"}).to_csv(
    queries_file, sep="\t", index=False
)

# Save full qrels
qrels_file = os.path.join("full", "qrels.tsv")
full_qrels.to_csv(qrels_file, sep="\t", index=False)

# Save full positives (query–passage pairs)
full_positives_df = pd.merge(
    full_qrels,
    queries_df[["query_id", "text"]].rename(columns={"text": "query"}),
    on="query_id",
).merge(
    corpus_df[["corpus_id", "text"]].rename(columns={"text": "passage"}),
    on="corpus_id",
    how="inner"
)

retained_full_qrels_count = len(full_positives_df)

full_positives_file = os.path.join("full", "positives.tsv")
full_positives_df[["query", "passage"]].rename(
    columns={"query": "sentence1", "passage": "sentence2"}
).to_csv(full_positives_file, sep="\t", index=False)

# Save full summary
full_summary_file = os.path.join("full", "summary.tsv")
full_positives_df[["query_id", "query", "corpus_id", "passage", "rel"]].to_csv(
    full_summary_file, sep="\t", index=False
)

print(f"Saved full dataset:")
print(f"   - Queries: {len(queries_df)}")
print(f"   - Corpus: {len(corpus_df)} (full corpus)")
print(f"   - Original Qrels: {original_full_qrels_count}")
print(f"   - Retained Qrels (after cleaning corpus): {retained_full_qrels_count}")
print(f"   - Positives: {len(full_positives_df)}")
print(f"   - Summary: {len(full_positives_df)}")


# ----------------------------
# Split queries into 80/10/10
# ----------------------------
print("\nSplitting dataset (80/10/10)...")
train_queries, temp_queries = train_test_split(
    queries_df, test_size=0.2, random_state=42, shuffle=True
)
val_queries, test_queries = train_test_split(
    temp_queries, test_size=0.5, random_state=42, shuffle=True
)

splits = {"train": train_queries, "val": val_queries, "test": test_queries}

# ----------------------------
# Save splits in train/val/test folders
# ----------------------------
for split_name, split_queries in splits.items():
    print(f"\nProcessing {split_name} split...")

    # Ensure split folder exists
    os.makedirs(split_name, exist_ok=True)

    # Filter qrels for this split queries - only include pairs that exist in cleaned queries AND corpus
    split_qrels = qrels_df[
        qrels_df["query_id"].isin(split_queries["query_id"]) &
        qrels_df["corpus_id"].isin(corpus_df["corpus_id"])
        ]
    original_qrels_count = len(split_qrels)

    # Save full corpus (not filtered)
    corpus_file = os.path.join(split_name, "corpus.tsv")
    corpus_df[["corpus_id", "text"]].to_csv(corpus_file, sep="\t", index=False)

    # Save queries
    queries_file = os.path.join(split_name, "queries.tsv")
    split_queries[["query_id", "text"]].rename(columns={"text": "query"}).to_csv(
        queries_file, sep="\t", index=False
    )

    # Save qrels
    qrels_file = os.path.join(split_name, "qrels.tsv")
    split_qrels.to_csv(qrels_file, sep="\t", index=False)

    # Save positives (query–passage pairs)
    positives_df = pd.merge(
        split_qrels,
        split_queries[["query_id", "text"]].rename(columns={"text": "query"}),
        on="query_id",
    ).merge(
        corpus_df[["corpus_id", "text"]].rename(columns={"text": "passage"}),
        on="corpus_id",
        how="inner"
    )

    retained_qrels_count = len(positives_df)

    positives_file = os.path.join(split_name, "positives.tsv")
    positives_df[["query", "passage"]].rename(
        columns={"query": "sentence1", "passage": "sentence2"}
    ).to_csv(positives_file, sep="\t", index=False)

    # Save summary
    summary_file = os.path.join(split_name, "summary.tsv")
    positives_df[["query_id", "query", "corpus_id", "passage", "rel"]].to_csv(
        summary_file, sep="\t", index=False
    )

    print(f"Saved {split_name} split:")
    print(f"   - Queries: {len(split_queries)}")
    print(f"   - Corpus: {len(corpus_df)} (full corpus)")
    print(f"   - Original Qrels: {original_qrels_count}")
    print(f"   - Retained Qrels (after cleaning corpus): {retained_qrels_count}")
    print(f"   - Positives: {len(positives_df)}")
    print(f"   - Summary: {len(positives_df)}")

print("\nAll splits processed successfully!")
