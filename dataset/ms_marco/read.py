import pandas as pd
import json
import textwrap

# Set pandas options to show full content
pd.set_option('display.max_colwidth', None)

# Read the first row from each TSV file
queries_df = pd.read_csv('docleaderboard-queries.tsv', sep='\t', nrows=1)
top100_df = pd.read_csv('docleaderboard-top100.tsv', sep='\t', nrows=1)
fulldocs_df = pd.read_csv('fulldocs.tsv', sep='\t', nrows=1)

def pretty_print(df, title, wrap_width=80):
    print(f"\n{'='*100}")
    print(f"{title.upper()} (first {len(df)} row{'s' if len(df) > 1 else ''})")
    print(f"{'='*100}\n")

    for i, row in df.iterrows():
        print(f"Row {i+1}:")
        print("{")
        for key, value in row.to_dict().items():
            key_str = json.dumps(str(key), ensure_ascii=False)
            val_str = json.dumps(str(value), ensure_ascii=False)

            # Wrap long lines for values
            wrapped_val = textwrap.fill(val_str, width=wrap_width,
                                        subsequent_indent=' ' * 6)
            print(f"  {key_str}: {wrapped_val}")
        print("}")
        print('-' * 100)

# Print all three
pretty_print(queries_df, 'docleaderboard-queries.tsv')
pretty_print(top100_df, 'docleaderboard-top100.tsv')
pretty_print(fulldocs_df, 'fulldocs.tsv')
