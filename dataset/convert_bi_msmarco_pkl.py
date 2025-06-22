import pandas as pd
import pickle
import os
import time

# === Config ===
output_dir = 'bi_marco_pkl'
sample_size_fulldocs = 10000  # Set None to load full fulldocs, or integer for sampling

os.makedirs(output_dir, exist_ok=True)

def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    print(f"{os.path.basename(filepath)} saved: {size_mb:.2f} MB")

def timed_read_csv(path, desc, col_names, limit_rows=None):
    start = time.time()
    df = pd.read_csv(path, sep='\t', header=None, names=col_names, quoting=3, nrows=limit_rows)
    df = df.dropna().reset_index(drop=True)
    df = df.apply(lambda col: col.map(lambda x: x.strip() if isinstance(x, str) else x))
    end = time.time()
    print(f"{desc} loaded: {df.shape[0]} rows, {df.shape[1]} columns in {end - start:.2f}s.")
    print(df.head(2))
    return df

total_start = time.time()

# === Load FULL data for queries and orcas queries ===
queries_df = timed_read_csv('ms_marco/docleaderboard-queries.tsv', 'Queries (Full)', ['qid', 'query'], None)

# === Load sampled fulldocs ===
fulldocs_df = timed_read_csv('ms_marco/fulldocs.tsv', 'Full Docs', ['url', 'title', 'body'], sample_size_fulldocs)
fulldocs_df['docid'] = fulldocs_df['url']
fulldocs_df['text'] = fulldocs_df['title'] + ' ' + fulldocs_df['body']

# === Load FULL data for qrels and top100 ===
top100_df = timed_read_csv('ms_marco/docleaderboard-top100.tsv', 'Top100 (Full)', ['qid', 'Q0', 'docid', 'rank', 'score', 'method'], None)

# === Save Pickle Files ===
save_pickle(queries_df, os.path.join(output_dir, 'queries.pkl'))
save_pickle(fulldocs_df, os.path.join(output_dir, 'fulldocs.pkl'))
save_pickle(top100_df, os.path.join(output_dir, 'top100.pkl'))

total_end = time.time()
print("\n✅ All MS MARCO pickle files created successfully.")
print(f"⏱ Total time taken: {total_end - total_start:.2f} seconds.")
