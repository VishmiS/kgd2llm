import pandas as pd
import json
import os

def convert_snli_tsv(snli_input_path, snli_output_path):
    column_names = ['Sentence1', 'Sentence2', 'Label']
    df = pd.read_csv(snli_input_path, sep='\t', header=None, names=column_names)
    df.to_csv(snli_output_path, sep='\t', index=False, header=False)
    print(f"SNLI converted and saved to {snli_output_path}")

def convert_multinli_jsonl_to_tsv(jsonl_path, tsv_output_path):
    records = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            records.append({
                'Sentence1': record.get('sentence1', ''),
                'Sentence2': record.get('sentence2', ''),
                'Label': record.get('gold_label', '')
            })
    df = pd.DataFrame(records)
    df.to_csv(tsv_output_path, sep='\t', index=False, header=False)
    print(f"MultiNLI converted and saved to {tsv_output_path}")

if __name__ == '__main__':
    # SNLI paths (adjust if necessary)
    snli_files = {
        'train':  ('snli_1.0/snli_1.0_train.tsv',  'snli_1.0/snli_1.0_train.tsv'),
        'dev':    ('snli_1.0/snli_1.0_dev.tsv',    'snli_1.0/snli_1.0_dev.tsv'),
        'test':   ('snli_1.0/snli_1.0_test.tsv',   'snli_1.0/snli_1.0_test.tsv')
    }

    # MultiNLI paths (JSONL → TSV)
    multinli_files = {
        'train':         ('multinli_1.0/multinli_1.0_train.jsonl',         'multinli_1.0/multinli_1.0_train.tsv'),
        'dev_matched':   ('multinli_1.0/multinli_1.0_dev_matched.jsonl',   'multinli_1.0/multinli_1.0_dev_matched.tsv'),
        'dev_mismatched':('multinli_1.0/multinli_1.0_dev_mismatched.jsonl','multinli_1.0/multinli_1.0_dev_mismatched.tsv'),
    }

    # Ensure output directories exist
    os.makedirs('snli_1.0', exist_ok=True)
    os.makedirs('multinli_1.0', exist_ok=True)

    # Convert SNLI
    for split, (input_path, output_path) in snli_files.items():
        convert_snli_tsv(input_path, output_path)

    # Convert MultiNLI
    for split, (input_path, output_path) in multinli_files.items():
        convert_multinli_jsonl_to_tsv(input_path, output_path)
