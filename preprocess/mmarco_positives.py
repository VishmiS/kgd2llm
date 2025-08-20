import os
import csv
import pickle
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Config ---------- #
MODEL_NAME = 'all-MiniLM-L6-v2'
DEVICE = 'cuda'
OUTPUT_DIR = '../outputs/pos_emb'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Utils ---------- #
def write_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"✅ Saved pickle: {file_path} (entries: {len(obj)})")

def load_mmarco_tsv_pairs(tsv_path):
    pairs = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            s1, s2 = row['sentence1'], row['sentence2']
            if isinstance(s1, str) and isinstance(s2, str):
                pairs.append((s1, s2))
    return pairs

def embed_texts_to_dict(texts, model, batch_size=64):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[i:i+batch_size]
        batch_emb = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        embeddings.extend(batch_emb)
    return embeddings

# ---------- Processing ---------- #
def process_mmarco_tsv(split_name, tsv_path, model):
    pairs = load_mmarco_tsv_pairs(tsv_path)
    print(f"\n▶ Loaded MS MARCO {split_name} pairs: {len(pairs)} samples")

    queries = [p[0] for p in pairs]
    positives = [p[1] for p in pairs]

    print(f"▶ Embedding MS MARCO {split_name} queries...")
    query_embs = embed_texts_to_dict(queries, model)
    print(f"▶ Embedding MS MARCO {split_name} positives...")
    pos_embs = embed_texts_to_dict(positives, model)

    similarity_dict = {}
    print(f"▶ Computing cosine similarities for {len(pairs)} pairs...")
    for (q, p), q_emb, p_emb in tqdm(zip(pairs, query_embs, pos_embs), total=len(pairs)):
        sim = cosine_similarity(
            np.array(q_emb).reshape(1, -1),
            np.array(p_emb).reshape(1, -1)
        )[0][0]
        similarity_dict[(q, p)] = [float(sim)]

    output_file = os.path.join(OUTPUT_DIR, f"mmarco_{split_name}_pos_emb.pkl")
    write_pickle(similarity_dict, output_file)

# ---------- Main ---------- #
def main():
    mmarco_paths = {
        "train_path": "../dataset/ms_marco/train/positives.tsv",
        "val_path": "../dataset/ms_marco/val/positives.tsv",
    }

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    for split in ['train', 'val']:
        process_mmarco_tsv(split, mmarco_paths[f"{split}_path"], model)

if __name__ == "__main__":
    main()
