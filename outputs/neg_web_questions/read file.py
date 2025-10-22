import pandas as pd
import collections

# ---------------- File paths ----------------
pkl_file = "/root/pycharm_semanticsearch/outputs/neg_web_questions/train/query_hard_negatives.pkl"
csv_file = "/root/pycharm_semanticsearch/outputs/neg_web_questions/web_questions_train_neg_long.csv"

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
                ans = f"'{ans}"
            records.append({"question": question, "hard_negative": ans})
else:
    raise ValueError(f"Unexpected pickle type: {type(data)}")

df_long = pd.DataFrame(records)
print(f"✅ Converted to DataFrame with {len(df_long)} rows and {len(df_long.columns)} columns")

# ---------------- Print full sentences for first 3 samples ----------------
print("Sample rows (full text):")
print(df_long.head(12).to_string(index=False))

# ---------------- Print number of unique queries ----------------
num_unique_queries = df_long['question'].nunique()
print(f"✅ Number of unique queries: {num_unique_queries}")

# ---------------- Optional: Truncate answers for CSV preview ----------------
df_long_trunc = df_long.copy()
df_long_trunc['hard_negative'] = df_long_trunc['hard_negative'].astype(str).str.slice(0, 200)  # e.g., first 200 chars

# ---------------- Save to CSV ----------------
df_long_trunc.to_csv(csv_file, index=False)
print(f"✅ CSV file created: {csv_file}")
