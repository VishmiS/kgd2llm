import pandas as pd
import collections

# ---------------- File paths ----------------
pkl_file = "/root/pycharm_semanticsearch/outputs/neg_mmarco/train/query_hard_negatives.pkl"
csv_file = "/root/pycharm_semanticsearch/outputs/neg_faiss/mmarco_train_query_hard_negatives.csv"

# ---------------- Load Pickle ----------------
data = pd.read_pickle(pkl_file)
print("✅ Loaded pickle file")
print("Type of data:", type(data))

# ---------------- Convert defaultdict to long-format DataFrame ----------------
records = []
if isinstance(data, (dict, collections.defaultdict)):
    for question, answers in data.items():
        for ans in answers:
            if isinstance(ans, str) and ans.startswith("="):
                ans = f"'{ans}"  # Prevent Excel misinterpretation if opened
            records.append({"question": question, "hard_negative": ans})
else:
    raise ValueError(f"Unexpected pickle type: {type(data)}")

df_long = pd.DataFrame(records)
print(f"✅ Converted to DataFrame with {len(df_long)} rows and {len(df_long.columns)} columns")

# ---------------- Print sample rows ----------------
print("Sample rows (full text):")
print(df_long.head(12).to_string(index=False))

# ---------------- Unique query count ----------------
num_unique_queries = df_long['question'].nunique()
print(f"✅ Number of unique queries: {num_unique_queries}")

# ---------------- Optional: Truncate long answers ----------------
df_long_trunc = df_long.copy()
df_long_trunc['hard_negative'] = df_long_trunc['hard_negative'].astype(str).str.slice(0, 200)

# ---------------- Save to CSV with header info ----------------
# with open(csv_file, "w", encoding="utf-8") as f:
#     f.write(f"# Number of unique queries: {num_unique_queries}\n")
#     df_long_trunc.to_csv(f, index=False)

print(f"✅ CSV file created with summary: {csv_file}")
