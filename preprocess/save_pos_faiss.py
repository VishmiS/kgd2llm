import os
import json
import csv
import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer


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


def load_snli_text_a(jsonl_path):
    queries = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['gold_label'] == 'entailment' and isinstance(row['sentence1'], str):
                queries.add(row['sentence1'])
    return list(queries)


def load_sts_text_a(csv_path):
    queries = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            score = float(row['score'])
            if score >= 4.0 and isinstance(row['sentence1'], str):
                queries.add(row['sentence1'])
    return list(queries)


def embed_texts_to_dict(texts, model, batch_size=64):
    embeddings_dict = {}
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[i:i+batch_size]
        embeddings = model.encode(batch, convert_to_tensor=False, show_progress_bar=False)
        for text, emb in zip(batch, embeddings):
            # Convert 384D embedding to 2D by taking first 2 elements (quick hack)
            emb_2d = emb[:2]  # or use PCA if you want something better
            embeddings_dict[text] = emb_2d.tolist()
    return embeddings_dict



# ---------- Processing ---------- #
def process_split(data_name, split, snli_paths, sts_paths, model):
    if data_name == 'snli':
        path = snli_paths[f"{split}_path"]
        texts = load_snli_text_a(path)
    elif data_name == 'sts':
        path = sts_paths[f"{split}_path"]
        texts = load_sts_text_a(path)
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")

    print(f"\n▶ Embedding {data_name} {split}: {len(texts)} samples")
    emb_dict = embed_texts_to_dict(texts, model)
    output_file = os.path.join(OUTPUT_DIR, f"{data_name}_{split}_pos_emb.pkl")
    write_pickle(emb_dict, output_file)


# ---------- Main ---------- #
def main():
    snli_paths = {
        "train_path": "../dataset/snli_1.0/snli_1.0_train.jsonl",
        "val_path": "../dataset/snli_1.0/snli_1.0_dev.jsonl",
    }
    sts_paths = {
        "train_path": "../dataset/sts/train.csv",
        "val_path": "../dataset/sts/validation.csv",
    }

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    for data_name in ['snli', 'sts']:
        for split in ['train', 'val']:
            process_split(data_name, split, snli_paths, sts_paths, model)


if __name__ == "__main__":
    main()
