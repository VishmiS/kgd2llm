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
    print(f"🔹 Loading {split} data...")
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

print("\n✅ All splits combined successfully.")

# ----------------------------
# Query length distribution
# ----------------------------
queries_df["query_length"] = queries_df["query"].astype(str).apply(lambda x: len(x.split()))
print(
    f"\n📊 Query length stats:"
    f"\n - min = {queries_df['query_length'].min()}"
    f"\n - max = {queries_df['query_length'].max()}"
    f"\n - mean = {queries_df['query_length'].mean():.2f}"
)

# ----------------------------
# Document length distribution
# ----------------------------
corpus_df["doc_length"] = corpus_df["text"].astype(str).apply(lambda x: len(x.split()))
print(
    f"\n📄 Document length stats:"
    f"\n - min = {corpus_df['doc_length'].min()}"
    f"\n - max = {corpus_df['doc_length'].max()}"
    f"\n - mean = {corpus_df['doc_length'].mean():.2f}"
)


# ----------------------------
# Corpus documents linked to multiple queries
# ----------------------------
# ----------------------------
# Qrels analysis
# ----------------------------
print(f"\n📋 Qrels DataFrame info:")
print(f"   Columns: {qrels_df.columns.tolist()}")
print(f"   Shape: {qrels_df.shape}")

# Check what columns are available and use appropriate ones
if not qrels_df.empty:
    # Try to identify the correct column names
    possible_doc_columns = ['corpus_id', 'doc_id', 'document_id', 'passage_id', 'text_id']
    possible_query_columns = ['query_id', 'question_id', 'qid']

    doc_col = None
    query_col = None

    # Find the actual column names
    for col in possible_doc_columns:
        if col in qrels_df.columns:
            doc_col = col
            break

    for col in possible_query_columns:
        if col in qrels_df.columns:
            query_col = col
            break

    if doc_col and query_col:
        print(f"   Using columns: '{doc_col}' for documents, '{query_col}' for queries")

        # Corpus documents linked to multiple queries
        doc_query_counts = qrels_df.groupby(doc_col)[query_col].nunique()
        multi_query_docs = doc_query_counts[doc_query_counts > 1]

        print(f"\n🔗 Corpus documents linked to multiple queries: {len(multi_query_docs)}")
        if len(multi_query_docs) > 0:
            print(f"   Example {doc_col}s linked to multiple queries:", multi_query_docs.head(5).index.tolist())

        # Additional qrels statistics
        print(f"\n📈 Qrels Statistics:")
        print(f"   Total query-document pairs: {len(qrels_df)}")
        print(f"   Unique queries: {qrels_df[query_col].nunique()}")
        print(f"   Unique documents: {qrels_df[doc_col].nunique()}")
        print(f"   Average documents per query: {len(qrels_df) / qrels_df[query_col].nunique():.2f}")

    else:
        print(f"❌ Could not identify document and query columns in qrels data")
        print(f"   Available columns: {qrels_df.columns.tolist()}")
else:
    print(f"❌ Qrels DataFrame is empty")
