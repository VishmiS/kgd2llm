import os
import csv
import pickle
from tqdm import tqdm

def convert_positives_tsv_to_pickle(tsv_path, output_pkl_path):
    """
    Converts a MS MARCO-style positives.tsv file into a pickle file.
    Stores all entries as a list of (query, passage) pairs or just passages.

    Args:
        tsv_path (str): Path to the input TSV file.
        output_pkl_path (str): Path to save the output pickle.
    """
    data = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc=f"Converting {os.path.basename(tsv_path)}"):
            if len(row) < 2:
                continue  # skip malformed lines
            query, passage = row[0].strip(), row[1].strip()
            # You can change the data structure here if your model expects something else
            data.append(passage)  # or data.append((query, passage))

    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"✅ Saved {len(data)} entries to: {output_pkl_path}")

# Example usage:
convert_positives_tsv_to_pickle("ms_marco/train/positives.tsv", "pkl/3mmarco/train_positives.pkl")
convert_positives_tsv_to_pickle("ms_marco/dev/positives.tsv", "pkl/3mmarco/dev_positives.pkl")
