import pandas as pd

# File paths (train set)
QUERIES_FILE = "/root/pycharm_semanticsearch/dataset/ms_marco/train/queries.tsv"
CORPUS_FILE  = "/root/pycharm_semanticsearch/dataset/ms_marco/train/corpus.tsv"
QRELS_FILE   = "/root/pycharm_semanticsearch/dataset/ms_marco/train/qrels.tsv"
OUTPUT_FILE  = "/root/pycharm_semanticsearch/dataset/ms_marco/train/MSMARCO_TRAIN_combined_truncated.xlsx"

# Load files
queries_df = pd.read_csv(QUERIES_FILE, sep="\t", names=["query_id", "query_text"], dtype=str)
corpus_df  = pd.read_csv(CORPUS_FILE, sep="\t", names=["passage_id", "passage_text"], dtype=str)
qrels_df   = pd.read_csv(QRELS_FILE, sep="\t", dtype=str)

# Drop unused columns if they exist (like 'zero')
if 'zero' in qrels_df.columns:
    qrels_df = qrels_df.drop(columns=['zero'])

# Strip whitespace from all relevant ID columns
queries_df['query_id'] = queries_df['query_id'].str.strip()
qrels_df['query_id']   = qrels_df['query_id'].str.strip()
qrels_df['passage_id'] = qrels_df['passage_id'].str.strip()
corpus_df['passage_id'] = corpus_df['passage_id'].str.strip()

# Merge qrels with queries
merged_df = qrels_df.merge(queries_df, on="query_id", how="left")

# Merge with corpus
merged_df = merged_df.merge(corpus_df, on="passage_id", how="left")

# Optional: check for missing passages
missing_passages = merged_df[merged_df['passage_text'].isnull()]
if not missing_passages.empty:
    print(f"⚠️ Missing {len(missing_passages)} passages after merge. Sample IDs:")
    print(missing_passages['passage_id'].unique()[:10])

# ------------------- NEW CODE START -------------------
# Clean up passage_text to prevent Excel formula interpretation
merged_df["passage_text"] = merged_df["passage_text"].astype(str).str.strip()
merged_df["passage_text"] = merged_df["passage_text"].apply(lambda x: f"'{x}" if x.startswith("=") else x)
# ------------------- NEW CODE END -------------------

# Truncate passage_text to 50 characters
merged_df["passage_text"] = merged_df["passage_text"].str.slice(0, 50)

# Reorder columns for clarity
merged_df = merged_df[["query_id", "query_text", "passage_id", "passage_text", "rel"]]

# Write only the Combined DataFrame to Excel
merged_df.to_excel(OUTPUT_FILE, sheet_name="Combined", index=False)

print(f"✅ Excel file created (Combined only, truncated passage text to 50 chars):\n📁 {OUTPUT_FILE}")
