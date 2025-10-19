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

def load_webq_tsv_pairs(tsv_path):
    """
    Load WebQuestions TSV and clean answers that accidentally include the question text again.
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
def process_webq_tsv(split_name, tsv_path, model):
    pairs = load_webq_tsv_pairs(tsv_path)
    print(f"\n▶ Loaded WebQuestions {split_name} pairs: {len(pairs)} samples")

    questions = [p[0] for p in pairs]
    answers = [p[1] for p in pairs]

    print(f"▶ Embedding WebQuestions {split_name} questions...")
    question_embs = embed_texts_to_list(questions, model)
    print(f"▶ Embedding WebQuestions {split_name} answers...")
    answer_embs = embed_texts_to_list(answers, model)

    # Load negatives TSV (assumes same format as positives, one negative per row)
    # Load hard negatives generated previously
    neg_pickle_path = f"../outputs/neg_web_questions/{split_name}/query_hard_negatives.pkl"
    with open(neg_pickle_path, 'rb') as f:
        neg_dict = pickle.load(f)  # {question: [neg1, neg2, ...]}

    # Embed negative answers
    neg_embs_dict = {}
    print(f"▶ Embedding WebQuestions {split_name} hard negatives...")
    for q, neg_list in tqdm(neg_dict.items(), desc="Embedding negatives"):
        if neg_list:
            neg_embs_dict[q] = embed_texts_to_list(neg_list, model)

    # Prepare teacher logits: ensure positive > negatives
    teacher_logits_dict = {}
    print(f"▶ Computing teacher logits for {len(pairs)} samples...")
    NEG_K = 8  # adjust if needed
    for i, ((q, a), q_emb, a_emb) in enumerate(tqdm(zip(pairs, question_embs, answer_embs), total=len(pairs))):
        pos_sim = cosine_similarity(np.array(q_emb).reshape(1, -1), np.array(a_emb).reshape(1, -1))[0][0]

        # select NEG_K negatives for this question
        neg_sims = []
        if q in neg_embs_dict:
            for neg_emb in neg_embs_dict[q]:
                sim = cosine_similarity(np.array(q_emb).reshape(1, -1), np.array(neg_emb).reshape(1, -1))[0][0]
                neg_sims.append(sim)

        # enforce positive > max negative
        if neg_sims:
            pos_sim = max(pos_sim, max(neg_sims) + 1e-3)

        logits = [pos_sim] + neg_sims
        teacher_logits_dict[(q.strip().lower(), a.strip().lower())] = logits

    output_file = os.path.join(OUTPUT_DIR, f"webq_{split_name}_teacher_logits.pkl")
    write_pickle(teacher_logits_dict, output_file)


# ---------- Main ---------- #
def main():
    webq_paths = {
        "train_path": "../dataset/web_questions/train/positives.tsv",
        "val_path": "../dataset/web_questions/val/positives.tsv",
    }

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)

    for split in ['train', 'val']:
        process_webq_tsv(split, webq_paths[f"{split}_path"], model)

if __name__ == "__main__":
    main()
