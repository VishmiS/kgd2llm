import pandas as pd
import pickle
import os

# Column names since TSVs have no header
column_names = ['Sentence1', 'Sentence2', 'Label']

# Paths for SNLI and MultiNLI train TSV files
snli_train_path = 'snli_1.0/snli_1.0_train.tsv'
mnli_train_path = 'multinli_1.0/multinli_1.0_train.tsv'

# Read SNLI and MultiNLI train data
snli_df = pd.read_csv(snli_train_path, sep='\t', header=None, names=column_names)
mnli_df = pd.read_csv(mnli_train_path, sep='\t', header=None, names=column_names)

# Combine datasets
combined_df = pd.concat([snli_df, mnli_df], ignore_index=True)

# Filter positive (entailment) and hard negative samples (non-entailment)
pos_samples = combined_df[combined_df['Label'] == 'entailment'][['Sentence1', 'Sentence2']].values.tolist()
hn_samples = combined_df[combined_df['Label'] != 'entailment'][['Sentence1', 'Sentence2']].values.tolist()

# Scores for the samples
pos_scores = [1.0] * len(pos_samples)
hn_scores = [0.0] * len(hn_samples)

# Output directory for combined pickles
output_dir = 'snli_multinli_combined_pkl'
os.makedirs(output_dir, exist_ok=True)

# Helper function to save pickle and print size
def save_pickle(obj, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    print(f"{os.path.basename(filepath)} saved: {size_mb:.2f} MB")

# Save combined pickles
save_pickle(pos_samples, os.path.join(output_dir, 'combined_pos.pkl'))
save_pickle(hn_samples, os.path.join(output_dir, 'combined_hn.pkl'))
save_pickle(pos_scores, os.path.join(output_dir, 'combined_pos_score.pkl'))
save_pickle(hn_scores, os.path.join(output_dir, 'combined_hn_score.pkl'))

print("All combined pickle files created successfully.")
