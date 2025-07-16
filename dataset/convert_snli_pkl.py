import pandas as pd
import os

def load_and_split_snli_tsv(tsv_path):
    # Match column names to those in your converted TSVs
    column_names = ['sentence1', 'sentence2', 'gold_label']
    df = pd.read_csv(tsv_path, sep='\t', header=0, names=column_names)  # header=0 if file has header line, else None

    # If your TSV has header row, use header=0; if no header row, use header=None

    # Split into entailment (positive) and others (negative)
    positives = df[df['gold_label'] == 'entailment']
    negatives = df[df['gold_label'] != 'entailment']
    return positives, negatives

def save_to_pkl(df, path):
    df.to_pickle(path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Saved to {path} ({len(df)} rows, {size_mb:.2f} MB)")

def print_first_rows(pkl_path, n=3):
    df = pd.read_pickle(pkl_path)
    print(f"\nFirst {n} rows of {pkl_path} (Total rows: {len(df)}):")
    print(df.head(n))

if __name__ == '__main__':
    # Input TSV paths — update these to your processed TSVs with headers
    snli_paths = {
        'train': 'processed/snli_1.0_train.tsv',
        'test':  'processed/snli_1.0_test.tsv',
        'val':   'processed/snli_1.0_dev.tsv'
    }

    # Output directory
    output_dir = '../dataset/pkl/1snli'
    os.makedirs(output_dir, exist_ok=True)

    for split, tsv_path in snli_paths.items():
        positives, negatives = load_and_split_snli_tsv(tsv_path)

        pos_path = os.path.join(output_dir, f'{split}_positives.pkl')
        neg_path = os.path.join(output_dir, f'{split}_negatives.pkl')

        save_to_pkl(positives, pos_path)
        save_to_pkl(negatives, neg_path)

        print_first_rows(pos_path)
        print_first_rows(neg_path)
