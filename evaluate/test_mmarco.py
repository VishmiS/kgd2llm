# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/evaluate/test_mmarco.py

import sys
import os
import ast
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import faiss
from argparse import Namespace
from collections import defaultdict
from model.pro_model import Mymodel
from tqdm import tqdm
import hashlib
import random, numpy as np, torch

# Paths
MODEL_PATH = "/root/pycharm_semanticsearch/PATH_TO_OUTPUT_MODEL/mmarco/final_student_model_fp32"
QUERIES_FILE = "/root/pycharm_semanticsearch/dataset/ms_marco/val/queries.tsv"
CORPUS_FILE  = "/root/pycharm_semanticsearch/dataset/ms_marco/val/corpus.tsv"
QRELS_FILE   = "/root/pycharm_semanticsearch/dataset/ms_marco/val/qrels.tsv"

# Settings
MAX_QUERIES = 30      # process all queries
MAX_CORPUS_DOCS = 1000   # process all corpus documents
RECALL_K = 10
BATCH_SIZE = 16
CORPUS_EMB_FILE = "corpus_embs.pt"

args = Namespace(
    num_heads=8,
    ln=True,
    norm=True,
    padding_side='right',
    neg_K=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_queries(max_queries=None):
    """Load queries from TSV"""
    queries = {}
    with open(QUERIES_FILE, "r") as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            if max_queries and i >= max_queries:
                break
            qid, qtext = line.strip().split("\t")
            queries[qid] = qtext
    return queries


def load_corpus(max_docs=None):
    """Load corpus from TSV"""
    corpus = {}
    with open(CORPUS_FILE, "r") as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            if max_docs and i >= max_docs:
                break
            doc_id, doc_text = line.strip().split("\t")
            corpus[doc_id] = doc_text
    return corpus


def load_qrels():
    """Load qrels from TSV"""
    qrels = defaultdict(set)
    with open(QRELS_FILE, "r") as f:
        next(f)  # skip header
        for line in f:
            qid, _, doc_id, rel = line.strip().split("\t")
            if rel != '1':
                continue
            qrels[qid].add(doc_id)
    return qrels


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_hash(model_path, corpus_file, max_docs, batch_size):
    """Generate a unique cache hash based on key parameters."""
    info = f"{model_path}-{corpus_file}-{max_docs}-{batch_size}"
    return hashlib.md5(info.encode()).hexdigest()

def encode_corpus(model, corpus, batch_size=BATCH_SIZE, force_rebuild=False):
    """
    Encode corpus embeddings and save to disk for reuse.
    Automatically rebuilds if cache is stale or invalid.
    """
    cache_hash = compute_hash(MODEL_PATH, CORPUS_FILE, len(corpus), batch_size)
    cache_file = f"corpus_embs_{cache_hash}.pt"

    # Optionally remove outdated default cache file
    if force_rebuild and os.path.exists(cache_file):
        print("[INFO] Rebuilding corpus embeddings (forced)...")
        os.remove(cache_file)

    # Load cache if it matches
    if os.path.exists(cache_file) and not force_rebuild:
        print(f"[INFO] Using cached embeddings: {cache_file}")
        data = torch.load(cache_file)
        return data["ids"], data["embs"].to(device)

    print("[INFO] Encoding corpus from scratch...")
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    corpus_embs_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Encoding Corpus"):
            batch_texts = corpus_texts[i:i + batch_size]
            batch_embs = model.encode(batch_texts, convert_to_tensor=True).to(device)
            corpus_embs_list.append(batch_embs)

    corpus_embs = torch.cat(corpus_embs_list, dim=0)
    torch.save({"ids": corpus_ids, "embs": corpus_embs.cpu()}, cache_file)
    print(f"[INFO] Saved new cache: {cache_file}")
    return corpus_ids, corpus_embs


def evaluate_mmarco():
    # Sanity checks
    for f in [QUERIES_FILE, CORPUS_FILE, QRELS_FILE, MODEL_PATH]:
        assert os.path.exists(f), f"{f} not found"

    set_seed(42)
    print("[INFO] Random seed fixed to 42 for reproducibility.")

    queries = load_queries(MAX_QUERIES)
    corpus = load_corpus(MAX_CORPUS_DOCS)
    qrels = load_qrels()

    model = Mymodel(model_name_or_path=MODEL_PATH, args=args)
    model.eval().to(device)

    # Encode corpus embeddings
    corpus_ids, corpus_embs = encode_corpus(model, corpus, force_rebuild=False)

    # FAISS CPU index (IP = inner product) with L2-normalized embeddings
    corpus_embs_np = corpus_embs.cpu().numpy().astype('float32')
    faiss.normalize_L2(corpus_embs_np)
    index_flat = faiss.IndexFlatIP(corpus_embs_np.shape[1])
    index_flat.add(corpus_embs_np)

    mrr_total, recall_total, num_eval = 0, 0, 0

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    with torch.no_grad():
        printed_examples = 0  # counter for printed samples
        for i in tqdm(range(0, len(query_texts), BATCH_SIZE), desc="Evaluating Queries"):
            batch_ids = query_ids[i:i + BATCH_SIZE]
            batch_texts = query_texts[i:i + BATCH_SIZE]

            # Encode batch queries
            batch_embs = model.encode(batch_texts, convert_to_tensor=True).cpu().numpy().astype('float32')
            faiss.normalize_L2(batch_embs)

            # FAISS CPU search
            D, I = index_flat.search(batch_embs, RECALL_K)

            for j, qid in enumerate(batch_ids):
                ranked_doc_ids = [corpus_ids[idx] for idx in I[j]]
                relevant = qrels.get(qid, set())

                # Compute MRR
                reciprocal_rank = 0
                for rank, doc_id in enumerate(ranked_doc_ids, start=1):
                    if doc_id in relevant:
                        reciprocal_rank = 1.0 / rank
                        break
                mrr_total += reciprocal_rank

                # Compute Recall@K
                recall_total += 1 if relevant & set(ranked_doc_ids) else 0

                # Print output only for the first 3 examples
                if printed_examples < 3:
                    print("\n--- Example ---")
                    print(f"Query   : {queries[qid]}")
                    print(f"Relevant: {list(relevant)[:3]} ...")
                    print(f"Top-{RECALL_K}: {ranked_doc_ids[:RECALL_K]}")
                    print(f"MRR     : {reciprocal_rank:.4f}")
                    print(f"Recall@{RECALL_K}: {bool(relevant & set(ranked_doc_ids))}")
                    printed_examples += 1

                num_eval += 1

    avg_mrr = mrr_total / num_eval
    avg_recall = recall_total / num_eval

    print(f"\nEval finished on {num_eval} queries")
    print(f"Avg MRR       : {avg_mrr:.4f}")
    print(f"Recall@{RECALL_K}: {avg_recall:.2%}")


if __name__ == "__main__":
    evaluate_mmarco()
