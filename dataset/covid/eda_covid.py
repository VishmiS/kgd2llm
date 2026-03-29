import os
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
splits = ["train", "val", "test"]


# ----------------------------
# Functions
# ----------------------------
def load_tsv(file_path):
    return pd.read_csv(file_path, sep="\t")


# ----------------------------
# Combine all splits
# ----------------------------
queries_list, corpus_list, qrels_list = [], [], []

for split in splits:
    print(f"Loading {split} data...")
    queries_path = os.path.join(split, "queries.tsv")
    corpus_path = os.path.join(split, "corpus.tsv")
    qrels_path = os.path.join(split, "qrels.tsv")

    if os.path.exists(queries_path):
        queries_list.append(load_tsv(queries_path))
    if os.path.exists(corpus_path):
        corpus_list.append(load_tsv(corpus_path))
    if os.path.exists(qrels_path):
        qrels_list.append(load_tsv(qrels_path))

# Merge all splits into single DataFrames
queries_df = pd.concat(queries_list, ignore_index=True)
corpus_df = pd.concat(corpus_list, ignore_index=True)
qrels_df = pd.concat(qrels_list, ignore_index=True)

print("\nAll splits combined successfully.")

# ----------------------------
# Query length distribution
# ----------------------------
queries_df["query_length"] = queries_df["query"].astype(str).apply(lambda x: len(x.split()))
print(
    f"\nQuery length stats:"
    f"\n - min = {queries_df['query_length'].min()}"
    f"\n - max = {queries_df['query_length'].max()}"
    f"\n - mean = {queries_df['query_length'].mean():.2f}"
)

# ----------------------------
# Document length distribution
# ----------------------------
corpus_df["doc_length"] = corpus_df["text"].astype(str).apply(lambda x: len(x.split()))
print(
    f"\nDocument length stats:"
    f"\n - min = {corpus_df['doc_length'].min()}"
    f"\n - max = {corpus_df['doc_length'].max()}"
    f"\n - mean = {corpus_df['doc_length'].mean():.2f}"
)


# ----------------------------
# Corpus documents linked to multiple queries
# ----------------------------
doc_query_counts = qrels_df.groupby("corpus_id")["query_id"].nunique()
multi_query_docs = doc_query_counts[doc_query_counts > 1]

print(f"\nCorpus documents linked to multiple queries: {len(multi_query_docs)}")
if len(multi_query_docs) > 0:
    print("   Example corpus_ids linked to multiple queries:", multi_query_docs.head(5).index.tolist())
