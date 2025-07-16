import pandas as pd
import json
import os

def convert_jsonl_to_tsv(jsonl_path, tsv_output_path, dataset_name):
    records = []
    print(f"🔄 Converting {dataset_name}: {jsonl_path}")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            gold_label = record.get('gold_label', '')
            if gold_label in ('-', ''):
                continue
            records.append({
                'sentence1': record.get('sentence1', ''),
                'sentence2': record.get('sentence2', ''),
                'gold_label': gold_label
            })

    df = pd.DataFrame(records)

    os.makedirs(os.path.dirname(tsv_output_path), exist_ok=True)
    df.to_csv(tsv_output_path, sep='\t', index=False, header=True)
    print(f"✅ Saved TSV to: {tsv_output_path}")
    print("➡️ Columns written:", df.columns.tolist())
    print(df.head(1), end='\n\n')


if __name__ == '__main__':
    snli_files = {
        'train': ('snli_1.0/snli_1.0_train.jsonl',  'snli_1.0/snli_1.0_train.tsv'),
        'val':   ('snli_1.0/snli_1.0_dev.jsonl',    'snli_1.0/snli_1.0_dev.tsv'),
        'test':  ('snli_1.0/snli_1.0_test.jsonl',   'snli_1.0/snli_1.0_test.tsv'),
    }

    multinli_files = {
        'train':          ('multinli_1.0/multinli_1.0_train.jsonl',          'multinli_1.0/multinli_1.0_train.tsv'),
        'dev_matched':    ('multinli_1.0/multinli_1.0_dev_matched.jsonl',    'multinli_1.0/multinli_1.0_dev_matched.tsv'),
        'dev_mismatched': ('multinli_1.0/multinli_1.0_dev_mismatched.jsonl', 'multinli_1.0/multinli_1.0_dev_mismatched.tsv'),
    }

    for split, (input_path, output_path) in snli_files.items():
        convert_jsonl_to_tsv(input_path, output_path, f"SNLI {split}")

    for split, (input_path, output_path) in multinli_files.items():
        convert_jsonl_to_tsv(input_path, output_path, f"MultiNLI {split}")
