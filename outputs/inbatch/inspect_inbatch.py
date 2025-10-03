import pickle

# 🔧 Set the path to your .pkl file here
pkl_path = "mmarco_train_inbatch.pkl"

def inspect_pickle(pkl_path):
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        print(f"\n✅ Successfully loaded: {pkl_path}")
        print(f"📦 Type of object: {type(data)}")

        if isinstance(data, dict):
            print(f"🔑 Dict keys (up to 10): {list(data.keys())[:2]}")
            print(f"\n🧪 Previewing first 2 key-value pairs:")
            for i, (k, v) in enumerate(data.items()):
                print(f"\n🔹 Record {i+1}")
                print(f"  Key: {k}")
                print(f"  Value: {v}")
                if i == 1:
                    break

        elif isinstance(data, list):
            print(f"📋 List length: {len(data)}")
            for i, item in enumerate(data[:2]):
                print(f"\n🔹 Record {i+1}")
                print(item)
        else:
            print(f"⚠️ Unhandled type preview:\n{data}")

    except Exception as e:
        print(f"❌ Failed to load pickle file: {e}")

inspect_pickle(pkl_path)
