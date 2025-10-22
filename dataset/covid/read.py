import pandas as pd

def print_first_n_records_parquet(path, n=1, max_len=500):
    """
    Print summary and first N records from a Parquet file.
    Shows total row count, columns, and truncated sample values.
    """
    print(f"\n--- Inspecting Parquet file: {path} ---")

    try:
        # Read parquet into DataFrame
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"❌ Error reading parquet file: {e}")
        return

    num_rows, num_cols = df.shape
    print(f"✅ Loaded Parquet file successfully!")
    print(f"➡️  Rows: {num_rows:,}")
    print(f"➡️  Columns: {num_cols}")
    print(f"➡️  Column names: {list(df.columns)}\n")

    if num_rows == 0:
        print("⚠️  File is empty — no records to display.")
        return

    # Display first N rows
    sample = df.head(n)
    print(f"--- Showing first {min(n, num_rows)} records ---")

    for i, row in enumerate(sample.itertuples(index=False), start=1):
        print(f"\nRecord {i}:")
        for col, val in zip(df.columns, row):
            val_str = str(val)
            if len(val_str) > max_len:
                val_str = val_str[:max_len] + "..."
            print(f"  {col}: {val_str}")

    print("\n--- End of sample ---")
    print(f"📊 Total rows in file: {num_rows:,}\n")


# ===============================
# Example usage:
# ===============================
print_first_n_records_parquet('/root/pycharm_semanticsearch/dataset/covid/train/queries.parquet')
print_first_n_records_parquet('/root/pycharm_semanticsearch/dataset/covid/dev/queries.parquet')
# print_first_n_records_parquet('/root/pycharm_semanticsearch/dataset/covid/train/qrels.parquet')
# print_first_n_records_parquet('/root/pycharm_semanticsearch/dataset/covid/docs/docs.parquet')
