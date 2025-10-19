# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/evaluate/test_mmarco.py

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import faiss
from argparse import Namespace
from collections import defaultdict
from model.pro_model import Mymodel
from tqdm import tqdm
import hashlib
import random, numpy as np, torch
import pandas as pd
import re
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import pandas as pd
from peft import get_peft_model, LoraConfig

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)

# Paths
MODEL_PATH = "/root/pycharm_semanticsearch/PATH_TO_OUTPUT_MODEL/mmarco/final_student_model_fp32"
QUERIES_FILE = "/root/pycharm_semanticsearch/dataset/ms_marco/val/queries.tsv"
CORPUS_FILE  = "/root/pycharm_semanticsearch/dataset/ms_marco/val/corpus.tsv"
QRELS_FILE   = "/root/pycharm_semanticsearch/dataset/ms_marco/val/qrels.tsv"

# Settings
MAX_QUERIES = 500     # process all queries
MAX_CORPUS_DOCS = 1008986   # process all corpus documents
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

    # Optionally remove outdated cache file
    if force_rebuild and os.path.exists(cache_file):
        print("[INFO] Rebuilding corpus embeddings (forced)...")
        os.remove(cache_file)

    # Load cache if it exists
    if os.path.exists(cache_file) and not force_rebuild:
        print(f"[INFO] Using cached embeddings: {cache_file}")
        data = torch.load(cache_file, map_location=device)
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

    # ✅ Normalize once after concatenation
    corpus_embs = corpus_embs / corpus_embs.norm(dim=-1, keepdim=True)
    print(f"[DEBUG] Corpus embedding norm (mean): {torch.norm(corpus_embs, dim=-1).mean().item():.4f}")

    # Save to disk
    torch.save({"ids": corpus_ids, "embs": corpus_embs.cpu()}, cache_file)
    print(f"[INFO] Saved L2-normalized corpus embeddings to {cache_file}")

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

    # 🔹 Filter queries to include only those that have at least one relevant doc
    queries = {qid: qtext for qid, qtext in queries.items() if qid in qrels}

    print(f"[INFO] Filtered queries to only those with qrels. Total unique queries with answers: {len(queries)}")

    # Initialize your student model
    model = Mymodel(model_name_or_path=MODEL_PATH, args=args).to(device)

    # Load the trained student weights safely (ignore missing/extra keys)
    state_dict = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.eval()  # set to eval mode

    # Encode corpus embeddings
    corpus_ids, corpus_embs = encode_corpus(model, corpus, force_rebuild=False)
    print(f"[DEBUG] Corpus embedding norm (mean): {torch.norm(corpus_embs, dim=-1).mean().item():.4f}")

    # FAISS CPU index (IP = inner product) with L2-normalized embeddings
    corpus_embs_np = corpus_embs.cpu().numpy().astype('float32')
    index_flat = faiss.IndexFlatIP(corpus_embs_np.shape[1])
    index_flat.add(corpus_embs_np)

    mrr_total, recall_total, num_eval = 0, 0, 0
    results = []

    query_ids = list(queries.keys())
    query_texts = [
        re.sub(r"^[\.\s]+", "", re.sub(r"[\s]+", " ", queries[qid].strip()))
        for qid in query_ids
    ]

    with torch.no_grad():
        printed_examples = 0  # counter for printed samples
        for i in tqdm(range(0, len(query_texts), BATCH_SIZE), desc="Evaluating Queries"):
            batch_ids = query_ids[i:i + BATCH_SIZE]
            batch_texts = query_texts[i:i + BATCH_SIZE]

            # Encode batch queries
            batch_embs = model.encode(batch_texts, convert_to_tensor=True)
            batch_embs = batch_embs / batch_embs.norm(dim=-1, keepdim=True)  # ✅ normalize in torch
            batch_embs = batch_embs.cpu().numpy().astype("float32")
            print(f"[DEBUG] Query embedding norm (mean): {np.linalg.norm(batch_embs, axis=1).mean():.4f}")

            # FAISS CPU search
            D, I = index_flat.search(batch_embs, RECALL_K)


            for j, qid in enumerate(batch_ids):
                ranked_doc_ids = [corpus_ids[idx] for idx in I[j]]
                relevant = qrels.get(qid, set())

                # --- EXTRA DEBUG: Check if relevant docs are even similar to the query ---
                if num_eval == 0:  # only print for first evaluated query to avoid flooding logs
                    q_emb = batch_embs[j]
                    relevant = qrels.get(qid, set())
                    print("\n================= ADVANCED DEBUG =================")
                    print(f"Query ID   : {qid}")
                    print(f"Query Text : {queries[qid]}")
                    print(f"Relevant Docs: {list(relevant)}")

                    # 1️⃣ Check cosine similarity of relevant docs
                    for rel_doc_id in relevant:
                        if rel_doc_id in corpus:
                            rel_idx = corpus_ids.index(rel_doc_id)
                            rel_emb = corpus_embs_np[rel_idx]
                            rel_sim = np.dot(q_emb, rel_emb)
                            print(f"CosSim(Query, {rel_doc_id}) = {rel_sim:.4f}")
                            print(f"Snippet: {corpus[rel_doc_id][:120]}...")
                        else:
                            print(f"[WARNING] Relevant doc {rel_doc_id} not found in corpus!")

                    # 2️⃣ Check average similarity to random 10 corpus docs
                    rand_sample = np.random.choice(len(corpus_ids), 10, replace=False)
                    avg_rand_sim = np.mean([np.dot(q_emb, corpus_embs_np[idx]) for idx in rand_sample])
                    print(f"Average cosine similarity to random corpus docs: {avg_rand_sim:.4f}")

                    # 3️⃣ Compare against top retrieved docs
                    print("\nTop-10 retrieved docs (with CosSim):")
                    for rank, idx in enumerate(I[j], start=1):
                        doc_id = corpus_ids[idx]
                        sim = np.dot(q_emb, corpus_embs_np[idx])
                        print(f"Rank {rank:02d}: {doc_id}, CosSim={sim:.4f}, Snippet: {corpus[doc_id][:100]}")

                    # 4️⃣ Summary reason hint
                    print("\n--- DIAGNOSTIC INTERPRETATION ---")
                    print("If relevant doc CosSim << top-10 CosSim → Embeddings are mismatched (model issue).")
                    print(
                        "If relevant doc CosSim ≈ top-10 CosSim but not retrieved → FAISS index or normalization issue.")
                    print("If relevant doc not found in corpus → Corpus–qrels mismatch.")
                    print("=================================================\n")



                # Compute MRR
                reciprocal_rank = 0
                for rank, doc_id in enumerate(ranked_doc_ids, start=1):
                    if doc_id in relevant:
                        reciprocal_rank = 1.0 / rank
                        break
                mrr_total += reciprocal_rank

                # Compute Recall@K
                recall_total += 1 if relevant & set(ranked_doc_ids) else 0

                # ------------------- Human-readable explanation -------------------
                # ------------------- Debug for first query only -------------------
                if num_eval == 0:  # only for the first query
                    rel_doc_id = list(relevant)[0]
                    rel_idx = corpus_ids.index(rel_doc_id)
                    rel_emb = corpus_embs_np[rel_idx]
                    cos_sim_rel = np.dot(batch_embs[j], rel_emb)
                    print(f"Cosine similarity with relevant doc {rel_doc_id}: {cos_sim_rel:.4f}")
                    print("\n--- DEBUG: First Query ---")
                    print(f"Query ID : {qid}")
                    print(f"Query Text: {queries[qid]}")
                    print(f"Relevant Doc IDs: {list(relevant)}")

                    # Check if relevant docs exist in corpus
                    for doc_id in relevant:
                        if doc_id not in corpus:
                            print(f"WARNING: Relevant doc {doc_id} not in corpus!")
                        else:
                            print(f"Relevant doc {doc_id} exists in corpus. Snippet: {corpus[doc_id][:100]}...")

                    # Check query embedding norm
                    q_emb = batch_embs[j]
                    print(f"Query embedding norm: {np.linalg.norm(q_emb):.4f}")

                    # Check corpus embedding norms of top 10 retrieved docs
                    print("Top-10 retrieved docs with cosine similarity:")
                    for rank, idx in enumerate(I[j], start=1):
                        doc_id = corpus_ids[idx]
                        doc_emb = corpus_embs_np[idx]
                        sim = np.dot(q_emb, doc_emb)
                        snippet = corpus[doc_id][:100]
                        print(f"Rank {rank}: Doc ID {doc_id}, CosSim={sim:.4f}, Snippet: {snippet}...")

                # Existing human-readable explanation
                top_doc_indices = I[j]
                ranked_docs_with_text = [(corpus_ids[idx], corpus[corpus_ids[idx]]) for idx in top_doc_indices]

                first_rel_rank = None
                for rank, doc_id in enumerate(ranked_doc_ids, start=1):
                    if doc_id in relevant:
                        first_rel_rank = rank
                        break

                if first_rel_rank:
                    explanation = f"First relevant document found at rank {first_rel_rank}."
                else:
                    explanation = "No relevant documents found in top-10 retrieval."

                results.append({
                    "query_id": qid,
                    "query_text": queries[qid],
                    "relevant_doc_ids": list(relevant),
                    f"top_{RECALL_K}_doc_ids": ranked_doc_ids[:RECALL_K],
                    "MRR": reciprocal_rank,
                    f"Recall@{RECALL_K}": 1 if relevant & set(ranked_doc_ids) else 0,
                    "Human_readable_explanation": explanation,
                    "Top_docs_sample": ranked_docs_with_text[:3]
                })

                # Print output only for the first 3 examples
                if printed_examples < 3:
                    print("\n--- Example ---")
                    print(f"Query   : {queries[qid]}")
                    print(f"Relevant: {list(relevant)[:3]} ...")
                    print(f"Top-{RECALL_K}: {ranked_doc_ids[:RECALL_K]}")
                    print(f"MRR     : {reciprocal_rank:.4f}")
                    print(f"Recall@{RECALL_K}: {bool(relevant & set(ranked_doc_ids))}")
                    print(f"Explanation: {explanation}")
                    print("Top 3 retrieved docs (ID: snippet):")
                    for doc_id, snippet in ranked_docs_with_text[:3]:
                        print(f"  {doc_id}: {snippet[:100]}...")
                    printed_examples += 1


                num_eval += 1

    avg_mrr = mrr_total / num_eval
    avg_recall = recall_total / num_eval

    print(f"\nEval finished on {num_eval} queries")
    print(f"Avg MRR       : {avg_mrr:.4f}")
    print(f"Recall@{RECALL_K}: {avg_recall:.4f}")

    df = pd.DataFrame(results)
    output_excel = f"mmarco_eval_results_top{RECALL_K}.xlsx"
    df.to_excel(output_excel, index=False)
    print(f"[INFO] Saved detailed results to: {output_excel}")

if __name__ == "__main__":
    evaluate_mmarco()
