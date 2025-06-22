import pandas as pd
import json

# Set display options to show full content without truncation
pd.set_option('display.max_colwidth', None)  # Show full column content
pd.set_option('display.max_rows', 5)         # Limit max rows shown

# Read first 5 rows from each JSONL file
queries_df = pd.read_json('multinli_1.0_train.jsonl', lines=True, nrows=1)
top100_df = pd.read_json('multinli_1.0_dev_matched.jsonl', lines=True, nrows=1)
qrels_df = pd.read_json('multinli_1.0_dev_mismatched.jsonl', lines=True, nrows=3)

def pretty_print(df, title):
    print(f"\n=== {title} (first 5 rows) ===\n")
    for i, row in df.iterrows():
        print(f"Row {i+1}:")
        print(json.dumps(row.to_dict(), indent=2, ensure_ascii=False))
        print('-' * 80)

pretty_print(queries_df, 'multinli_1.0_train.jsonl')
pretty_print(top100_df, 'multinli_1.0_dev_matched.jsonl')
pretty_print(qrels_df, 'multinli_1.0_dev_mismatched.jsonl')
