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
    Load MSMarco TSV and clean answers that accidentally include the question text again.
    """
    pairs = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            question = row['sentence1'].strip()
            answer = row['sentence2'].strip()

            # 🧹 If the answer starts with the question text, remove it
            if answer.lower().startswith(question.lower()):
                answer = answer[len(question):].strip(" ?")

            # only keep valid (q,a)
            if question and answer:
                pairs.append((question, answer))
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
    print(f"\n▶ Loaded MSMARCO {split_name} pairs: {len(pairs)} samples")

    questions = [p[0] for p in pairs]
    answers = [p[1] for p in pairs]

    print(f"▶ Embedding MSMARCO {split_name} questions...")
    question_embs = embed_texts_to_list(questions, model)
    print(f"▶ Embedding MSMARCO {split_name} answers...")
    answer_embs = embed_texts_to_list(answers, model)

    # Prepare teacher logits: ensure positive > negatives
    # Prepare positive embeddings dictionary (similar to inspect_positives.py output)
    pos_emb_dict = {}
    print(f"▶ Computing positive similarities for {len(pairs)} samples...")

    for i, ((q, a), q_emb, a_emb) in enumerate(tqdm(zip(pairs, question_embs, answer_embs), total=len(pairs))):
        sim = cosine_similarity(
            np.array(q_emb).reshape(1, -1),
            np.array(a_emb).reshape(1, -1)
        )[0][0]
        pos_emb_dict[(q.strip().lower(), a.strip().lower())] = [float(sim)]

    output_file = os.path.join(OUTPUT_DIR, f"mmarco_{split_name}_pos_emb.pkl")
    write_pickle(pos_emb_dict, output_file)


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
