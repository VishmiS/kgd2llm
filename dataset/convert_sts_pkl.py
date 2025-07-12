import pandas as pd
import os

# Input and output directories
csv_dir = "../dataset/sts"
pkl_dir = "../dataset/pkl/2sts"
os.makedirs(pkl_dir, exist_ok=True)

splits = ['train', 'validation', 'test']  # Adjust if your files use 'dev' instead of 'validation'

for split in splits:
    csv_path = os.path.join(csv_dir, f"{split}.csv")

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        continue

    # Load CSV with proper quoting to handle commas inside sentences
    try:
        df = pd.read_csv(csv_path, quotechar='"')
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        continue

    # Check columns loaded correctly
    expected_columns = {"split", "genre", "dataset", "year", "sid", "score", "sentence1", "sentence2"}
    if not expected_columns.issubset(df.columns):
        print(f"Warning: Columns missing or different in {csv_path}")
        print("Found columns:", df.columns.tolist())

    # Split into positives and negatives based on score threshold
    positives = df[df["score"] > 4.0]
    negatives = df[df["score"] <= 4.0]

    # Save positives
    pos_path = os.path.join(pkl_dir, f"{split}_positives.pkl")
    positives.to_pickle(pos_path)
    pos_size_mb = os.path.getsize(pos_path) / (1024 * 1024)
    print(f"{split}_positives.pkl: {pos_size_mb:.2f} MB")
    print(f"First 3 rows of {split}_positives.pkl:")
    print(positives.head(3))
    print()

    # Save negatives
    neg_path = os.path.join(pkl_dir, f"{split}_negatives.pkl")
    negatives.to_pickle(neg_path)
    neg_size_mb = os.path.getsize(neg_path) / (1024 * 1024)
    print(f"{split}_negatives.pkl: {neg_size_mb:.2f} MB")
    print(f"First 3 rows of {split}_negatives.pkl:")
    print(negatives.head(3))
    print("-" * 60)
