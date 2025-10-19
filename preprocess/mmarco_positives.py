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
    """
    Load MS MARCO TSV and clean passages that accidentally include the query text again.
    """
    pairs = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            query = row['sentence1'].strip()
            passage = row['sentence2'].strip()

            # 🧹 Clean up repeated query in passage
            if passage.lower().startswith(query.lower()):
                passage = passage[len(query):].strip(" ?")

            if query and passage:
                pairs.append((query, passage))
    return pairs


def embed_texts_to_list(texts, model, batch_size=64):
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
    query_embs = embed_texts_to_list(queries, model)

    print(f"▶ Embedding MS MARCO {split_name} positives...")
    pos_embs = embed_texts_to_list(positives, model)

    # ---------- Load FAISS negatives ---------- #
    neg_pickle_paths = {
        "train": "/root/pycharm_semanticsearch/outputs/neg_faiss/mmarco_train_neg.pkl",
        "val": "/root/pycharm_semanticsearch/outputs/neg_faiss/mmarco_val_neg.pkl",
    }

    neg_pickle_path = neg_pickle_paths[split_name]
    print(f"▶ Loading FAISS negatives from: {neg_pickle_path}")

    with open(neg_pickle_path, 'rb') as f:
        neg_dict = pickle.load(f)  # {query: [neg1, neg2, ...]}

    print(f"✅ Loaded {len(neg_dict)} negative sets for {split_name}")

    # ---------- Embed negative passages ---------- #
    neg_embs_dict = {}
    print(f"▶ Embedding MS MARCO {split_name} hard negatives...")
    for q, neg_list in tqdm(neg_dict.items(), desc="Embedding negatives"):
        if neg_list:
            neg_embs_dict[q] = embed_texts_to_list(neg_list, model)

    # ---------- Compute teacher logits ---------- #
    teacher_logits_dict = {}
    print(f"▶ Computing teacher logits for {len(pairs)} samples...")
    NEG_K = 8  # limit number of negatives per query if needed

    for (q, p), q_emb, p_emb in tqdm(zip(pairs, query_embs, pos_embs), total=len(pairs), desc="Computing logits"):
        pos_sim = cosine_similarity(np.array(q_emb).reshape(1, -1), np.array(p_emb).reshape(1, -1))[0][0]

        neg_sims = []
        if q in neg_embs_dict:
            for neg_emb in neg_embs_dict[q][:NEG_K]:
                sim = cosine_similarity(np.array(q_emb).reshape(1, -1), np.array(neg_emb).reshape(1, -1))[0][0]
                neg_sims.append(sim)

        if neg_sims:
            pos_sim = max(pos_sim, max(neg_sims) + 1e-3)

        logits = [pos_sim] + neg_sims
        teacher_logits_dict[(q.strip().lower(), p.strip().lower())] = logits

    output_file = os.path.join(OUTPUT_DIR, f"mmarco_{split_name}_teacher_logits.pkl")
    write_pickle(teacher_logits_dict, output_file)


# ---------- Main ---------- #
def main():
    mmarco_paths = {
        "train": "../dataset/ms_marco/train/positives.tsv",
        "val": "../dataset/ms_marco/val/positives.tsv",
    }

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    for split in ['train', 'val']:
        process_mmarco_tsv(split, mmarco_paths[split], model)


if __name__ == "__main__":
    main()
