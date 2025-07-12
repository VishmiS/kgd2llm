import json

def print_first_n_records_unstructured(path, n=10):
    print(f"\n--- Showing first {n} unstructured records from file: {path} ---")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, list):
        # Case: Top-level JSON is a list
        for i, item in enumerate(data[:n]):
            val_str = json.dumps(item, indent=2)
            if len(val_str) > 500:
                val_str = val_str[:500] + "..."
            print(f"\nRecord {i+1}:\n{val_str}")
    elif isinstance(data, dict):
        # Case: Top-level JSON is a dict
        for i, (key, val) in enumerate(data.items()):
            if i >= n:
                break
            val_str = json.dumps(val, indent=2)
            if len(val_str) > 500:
                val_str = val_str[:500] + "..."
            print(f"\nKey: {key}\nValue:\n{val_str}")
    else:
        print("Unsupported JSON format (not a list or dict at the top level)")

# Usage example
# print_first_n_records_unstructured('train_v2.1.json')
print_first_n_records_unstructured('dev_v2.1.json')
# print_first_n_records_unstructured('eval_v2.1_public.json')
