import pickle
import os


def load_pickle(file_path):
    """Load a pickle file and return its content or None if error occurs."""
    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load pickle: {e}")
        return None


def inspect_logits(file_path, label="", neg_K=8):
    """Inspect the structure and some sample data of the logits pickle file."""
    print(f"\nInspecting {label} file: {file_path}")

    logits = load_pickle(file_path)
    if logits is None:
        return

    print(f"Type of logits: {type(logits)}")

    if isinstance(logits, list):
        print(f"List length: {len(logits)}")
        if logits:
            print(f"Sample logits[0]: {logits[0]}")
            print(f"Type of logits[0]: {type(logits[0])}")
        else:
            print("Empty list.")

    elif isinstance(logits, dict):
        print("Found dict. Inspecting sample keys and values:")
        for i, (key, val) in enumerate(logits.items()):
            print(f"  Key: {repr(key)[:100]}...")
            print(f"     Value type: {type(val)} | Value preview: {str(val)[:300]}")
            if isinstance(val, list) and val and isinstance(val[0], tuple):
                print(f"     Extracted score sample: {val[0][1]}")
            if i >= 2:
                break

        # Extra statistics for hard negatives
        lengths = [len(v) for v in logits.values()]
        num_zero = sum(1 for l in lengths if l == 0)
        num_too_few = sum(1 for l in lengths if l < neg_K)
        num_enough = sum(1 for l in lengths if l >= neg_K)

        print("\nHard Negative Stats:")
        print(f"  Total queries: {len(lengths)}")
        print(f"  Min hard negatives per query: {min(lengths)}")
        print(f"  Max hard negatives per query: {max(lengths)}")
        print(f"  Average hard negatives per query: {sum(lengths) / len(lengths):.2f}")
        print(f"  Queries with 0 hard negatives: {num_zero}")
        print(f"  Queries with < {neg_K} hard negatives: {num_too_few}")
        print(f"  Queries with ≥ {neg_K} hard negatives: {num_enough}")

    else:
        print(f"Unexpected type: {type(logits)}")
        print(f"Content preview: {str(logits)[:200]}")


if __name__ == "__main__":
    # Set your file paths here
    neg_file = "mmarco_train_neg.pkl"
    inspect_logits(neg_file, label="NEG", neg_K=8)
