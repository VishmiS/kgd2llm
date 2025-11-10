import pandas as pd
import os

def show_first_n_tsv_table(path, n=3, max_len=100):
    """
    Display the first N records of a TSV file in table format.
    Truncates long text fields for readability.
    """
    print(f"\n--- Inspecting TSV file: {path} ---")

    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    try:
        df = pd.read_csv(path, sep='\t')
    except Exception as e:
        print(f"❌ Error reading TSV file: {e}")
        return

    if df.empty:
        print("⚠️ File is empty — no records to display.")
        return

    # Truncate long text fields
    df_trunc = df.head(n).copy()
    for col in df_trunc.columns:
        df_trunc[col] = df_trunc[col].astype(str).apply(lambda x: x[:max_len] + '...' if len(x) > max_len else x)

    print(f"✅ First {min(n, len(df_trunc))} records:")
    print(df_trunc.to_string(index=False))
    print(f"\n📊 Total rows in file: {len(df):,}\n")


# ===============================
# Example usage for train folder
# ===============================
base_path = '/root/pycharm_semanticsearch/dataset/ms_marco/test'

for fname in ['queries.tsv', 'qrels.tsv', 'corpus.tsv', 'positives.tsv']:
    show_first_n_tsv_table(os.path.join(base_path, fname))
