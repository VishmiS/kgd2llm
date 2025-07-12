# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/preprocess/save_hardneg_faiss.py

import os
import json
import csv
import pickle
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, util


# Load pre-trained model once (outside the loop)
model = SentenceTransformer('all-MiniLM-L6-v2')  # fast, ~384-dim embeddings


def write_pickle(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def load_snli_en(jsonl_path):
    queries = []
    corpus = []
    pos_sample_dict = defaultdict(list)
    global_pos = set()

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            text_a = row['sentence1']
            text_b = row['sentence2']
            label = row['gold_label']

            if isinstance(text_b, str):
                corpus.append(text_b)

            if label == 'entailment':
                if isinstance(text_a, str):
                    queries.append(text_a)
                pos_sample_dict[text_a].append(text_b)
                global_pos.add(text_b)

    return queries, list(set(corpus)), pos_sample_dict, global_pos

def load_sts_csv(csv_path):
    queries = []
    corpus = []
    pos_sample_dict = defaultdict(list)
    global_pos = set()

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text_a = row['sentence1']
            text_b = row['sentence2']
            label = float(row['score'])

            if isinstance(text_b, str):
                corpus.append(text_b)

            if label >= 4.0:  # same as entailment-like threshold
                if isinstance(text_a, str):
                    queries.append(text_a)
                pos_sample_dict[text_a].append(text_b)
                global_pos.add(text_b)

    return queries, list(set(corpus)), pos_sample_dict, global_pos

def embed_texts(texts, model, batch_size=256):
    embeddings = []
    for start_idx in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
        batch = texts[start_idx:start_idx+batch_size]
        emb = model.encode(batch, convert_to_tensor=True, device='cuda', show_progress_bar=False)
        emb_cpu = emb.cpu().numpy()  # Move to CPU and convert to numpy
        embeddings.append(emb_cpu)
    embeddings = np.vstack(embeddings).astype('float32')
    return embeddings

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity if vectors normalized)
    # Move to GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index.add(embeddings)
    return gpu_index

def filter_similar_negatives(query, candidates, threshold=0.7):
    """
    Filter out negatives that are too semantically similar to the query.
    :param query: The anchor/query sentence (string)
    :param candidates: A list of candidate negatives (strings)
    :param threshold: Cosine similarity above which to discard
    :return: A filtered list of negative sentences
    """
    query_emb = model.encode(query, convert_to_tensor=True, device='cuda')
    cand_embs = model.encode(candidates, convert_to_tensor=True, device='cuda')

    # Compute cosine similarity
    similarities = util.cos_sim(query_emb, cand_embs)[0]

    # Filter: keep only those with similarity < threshold
    filtered = [
        cand for cand, sim in zip(candidates, similarities)
        if sim < threshold
    ]
    return filtered

def find_hard_negatives(
    queries,
    corpus,
    pos_sample_dict,
    model,
    query_embeddings,
    corpus_embeddings,
    K=10,
    global_pos=None
):
    print("Filtering corpus to remove positives and queries themselves...")

    # Build exclusion set
    all_exclude = set(queries)
    if global_pos:
        all_exclude.update(global_pos)
    exclude_norm = set(x.strip().lower() for x in all_exclude)

    # Build filtered corpus and embeddings together
    keep_indices = [i for i, c in enumerate(corpus) if c.strip().lower() not in exclude_norm]
    filtered_corpus = [corpus[i] for i in keep_indices]
    filtered_corpus_embeddings = corpus_embeddings[keep_indices]

    print(f"Filtered corpus size: {len(filtered_corpus)} (original: {len(corpus)})")

    # Normalize for cosine similarity
    faiss.normalize_L2(filtered_corpus_embeddings)
    faiss.normalize_L2(query_embeddings)

    print("Building FAISS index on GPU...")
    index = build_faiss_index(filtered_corpus_embeddings)

    print(f"Searching top {K+20} neighbors for each query...")
    distances, indices = index.search(query_embeddings, K + 20)

    hard_neg_sample_dict = defaultdict(list)

    print("Filtering hard negatives with semantic similarity...")
    for i, query in enumerate(tqdm(queries, desc="Filtering hard negatives")):
        candidate_texts = []
        seen = set()

        norm_query = str(query).strip().lower()
        norm_pos_set = set(x.strip().lower() for x in pos_sample_dict.get(query, []))

        for idx in indices[i]:
            candidate = filtered_corpus[idx]
            norm_candidate = candidate.strip().lower()

            if (
                norm_candidate == norm_query
                or norm_candidate in norm_pos_set
                or norm_candidate in seen
            ):
                continue

            seen.add(norm_candidate)
            candidate_texts.append(candidate)

            if len(candidate_texts) >= K * 3:
                break

        if not candidate_texts:
            continue

        # Vectorized similarity filtering (FASTER)
        query_emb = query_embeddings[i:i+1]  # Shape (1, dim)
        cand_embs = filtered_corpus_embeddings[indices[i]]
        sims = util.cos_sim(query_emb, cand_embs)[0]

        filtered = [
            cand for cand, sim in zip(candidate_texts, sims)
            if sim < 0.7
        ]

        hard_neg_sample_dict[query] = filtered[:K]

    return hard_neg_sample_dict


def compute_positive_logits(queries, pos_sample_dict, model):
    pos_logits_dict = defaultdict(list)
    for query in tqdm(queries, desc="Computing positive logits"):
        query_emb = model.encode([query], convert_to_tensor=True, device='cuda', show_progress_bar=False)
        pos_texts = pos_sample_dict.get(query, [])
        if not pos_texts:
            continue
        pos_embs = model.encode(pos_texts, convert_to_tensor=True, device='cuda', show_progress_bar=False)

        # Normalize embeddings for cosine similarity
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        pos_embs = pos_embs / np.linalg.norm(pos_embs, axis=1, keepdims=True)

        sims = np.dot(pos_embs, query_emb.T).flatten()
        pos_logits_dict[query] = list(zip(pos_texts, sims.tolist()))
    return pos_logits_dict

def process_dataset(data_name, split, snli_paths, sts_paths, K=10):
    output_dir = "outputs/neg_faiss"
    os.makedirs(output_dir, exist_ok=True)

    if data_name == 'snli':
        path = snli_paths[f"{split}_path"]
        queries, corpus, pos_sample_dict, global_pos = load_snli_en(path)
        output_pickle_neg = os.path.join(output_dir, f'snli_{split}_neg.pkl')
        output_pickle_pos = os.path.join(output_dir, f'snli_{split}_pos.pkl')

    elif data_name == 'sts':
        path = sts_paths[f"{split}_path"]
        queries, corpus, pos_sample_dict, global_pos = load_sts_csv(path)
        output_pickle_neg = os.path.join(output_dir, f'sts_{split}_neg.pkl')
        output_pickle_pos = os.path.join(output_dir, f'sts_{split}_pos.pkl')
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")

    print(f"Processing {data_name} {split} split, data size: queries={len(queries)}, corpus={len(corpus)}")

    # Load model once here
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    # Call the function to get hard negatives
    query_embeddings = embed_texts(queries, model)
    corpus_embeddings = embed_texts(corpus, model)

    hard_neg_sample_dict = find_hard_negatives(
        queries,
        corpus,
        pos_sample_dict,
        model,
        query_embeddings,
        corpus_embeddings,
        K,
        global_pos
    )

    # Compute positive logits for each query
    # pos_logits_dict = compute_positive_logits(queries, pos_sample_dict, model)

    # TO GENERATE dict[str, List[Tuple[str, float]]]
    # 📌 Save only the plain hard negatives (no logits)
    write_pickle(hard_neg_sample_dict, output_pickle_neg)
    # write_pickle(pos_logits_dict, output_pickle_pos)
    print(f"Saved hard negatives to {output_pickle_neg}")
    print(f"Saved positive logits to {output_pickle_pos}\n")

    # TO GENERATE list[float]
    # ✅ Extract logits only (List[float]) for saving
    neg_logits_flat = [0.0 for pairs in hard_neg_sample_dict.values() for _ in pairs]
    # pos_logits_flat = [score for pairs in pos_logits_dict.values() for (_, score) in pairs]

    # ✅ Save flat lists instead of defaultdicts
    neg_logits_path = output_pickle_neg.replace(".pkl", "_logits.pkl")
    pos_logits_path = output_pickle_pos.replace(".pkl", "_logits.pkl")

    write_pickle(neg_logits_flat, neg_logits_path)
    # write_pickle(pos_logits_flat, pos_logits_path)

    print(f"✅ Saved {len(neg_logits_flat)} negative logits to {neg_logits_path}")
    # print(f"✅ Saved {len(pos_logits_flat)} positive logits to {pos_logits_path}\n")



def main():
    # Paths for SNLI
    snli_paths = {
        "train_path": "dataset/snli_1.0/snli_1.0_train.jsonl",
        "val_path": "dataset/snli_1.0/snli_1.0_dev.jsonl",
    }

    # Paths for STS-B
    sts_paths = {
        "train_path": "dataset/sts/train.csv",
        "val_path": "dataset/sts/validation.csv",
    }

    K = 10  # top K negatives

    # Run SNLI train and val
    process_dataset('snli', 'train', snli_paths, sts_paths, K)
    process_dataset('snli', 'val', snli_paths, sts_paths, K)

    # Run STS train and val
    process_dataset('sts', 'train', snli_paths, sts_paths, K)
    process_dataset('sts', 'val', snli_paths, sts_paths, K)

if __name__ == "__main__":
    main()
