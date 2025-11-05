import pandas as pd
import os
import pickle
from collections import defaultdict


# ----------------------------
# Function to convert specific PKL file to XLSX
# ----------------------------
def convert_specific_pkl_to_xlsx(pkl_file_path):
    """
    Converts a specific PKL file to XLSX format.
    Handles different data types that might be stored in the PKL file.
    """
    if not os.path.exists(pkl_file_path):
        print(f"❌ File '{pkl_file_path}' not found!")
        return

    if not pkl_file_path.endswith(".pkl"):
        print(f"❌ File '{pkl_file_path}' is not a PKL file!")
        return

    xlsx_path = pkl_file_path.replace(".pkl", ".xlsx")

    try:
        # Read PKL file
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"📊 Loaded data type: {type(data)}")

        # Handle different data types
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, defaultdict):
            # Convert defaultdict to DataFrame
            df = convert_defaultdict_to_dataframe(data)
        elif isinstance(data, dict):
            # Convert regular dictionary to DataFrame
            df = convert_dict_to_dataframe(data)
        elif isinstance(data, list):
            # Convert list to DataFrame
            df = pd.DataFrame(data)
        else:
            print(f"❌ Unsupported data type in PKL file: {type(data)}")
            print(
                f"📋 Sample data: {list(data.items())[:3] if hasattr(data, 'items') else data[:3] if hasattr(data, '__getitem__') else 'Cannot display sample'}")
            return

        # Save as XLSX
        df.to_excel(xlsx_path, index=False)
        print(f"✅ Converted {pkl_file_path} → {xlsx_path}")
        print(f"📈 DataFrame shape: {df.shape}")
        print(f"📝 Columns: {list(df.columns)}")

    except Exception as e:
        print(f"❌ Error converting {pkl_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()


def convert_defaultdict_to_dataframe(default_dict):
    """
    Convert a defaultdict to a pandas DataFrame.
    """
    # First, let's inspect the structure
    sample_key = next(iter(default_dict.keys())) if default_dict else None
    sample_value = default_dict[sample_key] if sample_key else None

    print(f"🔍 Sample key: {sample_key}")
    print(f"🔍 Sample value type: {type(sample_value)}")

    # Try different conversion strategies based on the data structure
    if isinstance(sample_value, (list, tuple)):
        # If values are lists/tuples, create a row for each key-value pair
        rows = []
        for key, values in default_dict.items():
            if isinstance(values, (list, tuple)):
                for i, value in enumerate(values):
                    rows.append({'query': key, 'negative': value, 'index': i})
            else:
                rows.append({'query': key, 'negative': values})
        return pd.DataFrame(rows)

    elif isinstance(sample_value, dict):
        # If values are dictionaries, flatten them
        rows = []
        for key, value_dict in default_dict.items():
            if isinstance(value_dict, dict):
                row = {'query': key}
                row.update(value_dict)
                rows.append(row)
            else:
                rows.append({'query': key, 'value': value_dict})
        return pd.DataFrame(rows)

    else:
        # Simple key-value pairs
        return pd.DataFrame(list(default_dict.items()), columns=['query', 'negatives'])


def convert_dict_to_dataframe(regular_dict):
    """
    Convert a regular dictionary to a pandas DataFrame.
    """
    # Similar logic as above but for regular dict
    sample_key = next(iter(regular_dict.keys())) if regular_dict else None
    sample_value = regular_dict[sample_key] if sample_key else None

    print(f"🔍 Sample key: {sample_key}")
    print(f"🔍 Sample value type: {type(sample_value)}")

    if isinstance(sample_value, (list, tuple)):
        rows = []
        for key, values in regular_dict.items():
            if isinstance(values, (list, tuple)):
                for i, value in enumerate(values):
                    rows.append({'query': key, 'negative': value, 'index': i})
            else:
                rows.append({'query': key, 'negative': values})
        return pd.DataFrame(rows)

    elif isinstance(sample_value, dict):
        rows = []
        for key, value_dict in regular_dict.items():
            if isinstance(value_dict, dict):
                row = {'query': key}
                row.update(value_dict)
                rows.append(row)
            else:
                rows.append({'query': key, 'value': value_dict})
        return pd.DataFrame(rows)

    else:
        return pd.DataFrame(list(regular_dict.items()), columns=['query', 'negatives'])


# ----------------------------
# Convert specific PKL file
# ----------------------------

# Specify the full path to your PKL file
pkl_path = "neg_covid/train/query_hard_negatives.pkl"


# First, let's inspect what's in the file without converting
def inspect_pkl_file(pkl_file_path):
    """Inspect the contents of the PKL file to understand its structure."""
    try:
        with open(pkl_file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"🔍 Data type: {type(data)}")
        print(f"🔍 Number of items: {len(data) if hasattr(data, '__len__') else 'N/A'}")

        if hasattr(data, 'items'):
            sample_items = list(data.items())[:3]
            print("🔍 Sample items:")
            for key, value in sample_items:
                print(f"   Key: {key}")
                print(f"   Value type: {type(value)}")
                print(f"   Value: {value}")
                print("   ---")

        return data
    except Exception as e:
        print(f"❌ Error inspecting file: {str(e)}")
        return None


# First inspect the file
print("🔍 Inspecting PKL file structure...")
inspect_pkl_file(pkl_path)

print("\n🔄 Converting PKL file to XLSX...")
convert_specific_pkl_to_xlsx(pkl_path)