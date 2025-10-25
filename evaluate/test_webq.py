# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/evaluate/test_webq.py

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
MODEL_PATH = "/root/pycharm_semanticsearch/PATH_TO_OUTPUT_MODEL/webq/checkpoint-epoch-5"
QUERIES_FILE = "/root/pycharm_semanticsearch/dataset/web_questions/test/queries.tsv"
CORPUS_FILE  = "/root/pycharm_semanticsearch/dataset/web_questions/test/corpus.tsv"
QRELS_FILE   = "/root/pycharm_semanticsearch/dataset/web_questions/test/qrels.tsv"

# Settings
MAX_QUERIES = 10000     # process all queries
MAX_CORPUS_DOCS = 10000   # process all corpus documents
RECALL_K = 10
BATCH_SIZE = 16
CORPUS_EMB_FILE = "corpus_embs_webq.pt"

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
    """Generate a unique cache hash based on key parameters including model weights."""
    # Include model file modification time and size in hash
    model_file = f"{model_path}/pytorch_model_gpt_backup.bin"
    model_info = ""
    if os.path.exists(model_file):
        stat = os.stat(model_file)
        model_info = f"{stat.st_mtime}-{stat.st_size}"

    info = f"{model_path}-{model_info}-{corpus_file}-{max_docs}-{batch_size}"
    return hashlib.md5(info.encode()).hexdigest()


def encode_corpus(model, corpus, batch_size=BATCH_SIZE, force_rebuild=False):
    """
    Encode corpus embeddings and cache them with alignment verification.
    Auto-rebuilds cache if misaligned or stale.
    """
    import hashlib, json

    # Compute corpus hash
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    corpus_hash = hashlib.md5("\n".join(corpus_texts).encode("utf-8")).hexdigest()

    # Compute unique cache file name based on model and corpus info
    cache_hash = compute_hash(MODEL_PATH, CORPUS_FILE, len(corpus), batch_size)
    cache_file = f"corpus_embs_webq_{cache_hash}.pt"
    meta_file = f"{cache_file}.meta.json"

    # Try loading existing cache
    if os.path.exists(cache_file) and not force_rebuild:
        try:
            data = torch.load(cache_file, map_location=device)
            meta = json.load(open(meta_file)) if os.path.exists(meta_file) else {}
            if meta.get("corpus_hash") == corpus_hash and meta.get("num_docs") == len(corpus):
                print(f"[INFO] ✅ Cache verified and loaded: {cache_file}")
                return data["ids"], data["embs"].to(device)
            else:
                print("[WARNING] ⚠️ Cache hash mismatch — rebuilding embeddings...")
        except Exception as e:
            print(f"[WARNING] Failed to load cache ({e}), rebuilding embeddings...")

    # Encode corpus embeddings from scratch
    print("[INFO] Encoding corpus from scratch...")
    corpus_embs_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Encoding Corpus"):
            batch_texts = corpus_texts[i:i+batch_size]
            batch_embs = model.encode(batch_texts, convert_to_tensor=True).to(device)
            corpus_embs_list.append(batch_embs)

    corpus_embs = torch.cat(corpus_embs_list, dim=0)
    corpus_embs = corpus_embs / (corpus_embs.norm(dim=-1, keepdim=True) + 1e-12)

    # Save embeddings and aligned IDs
    torch.save({"ids": corpus_ids, "embs": corpus_embs.cpu()}, cache_file)
    meta_data = {
        "corpus_hash": corpus_hash,
        "num_docs": len(corpus_ids),
        "ids_sample": corpus_ids[:10],
        "emb_shape": list(corpus_embs.shape),
        "order_hash": hashlib.md5("".join(corpus_ids).encode()).hexdigest()
    }
    with open(meta_file, "w") as f:
        json.dump(meta_data, f, indent=2)

    print(f"[INFO] 💾 Corpus embeddings cached with verification at {cache_file}")
    return corpus_ids, corpus_embs


def map_parameter_names(state_dict):
    """Map parameter names from saved format to current model format"""
    mapping = {
        # Add mappings based on your model architecture
        # Example: 'plm_model.wte.weight': 'plm_model.embeddings.word_embeddings.weight',
        # You'll need to figure out the exact mapping for your model
    }

    new_state_dict = {}
    for old_name, param in state_dict.items():
        new_name = mapping.get(old_name, old_name)
        new_state_dict[new_name] = param

    return new_state_dict


def load_model_with_weights(model_path, args, device):
    """Load model with proper weight initialization - simplified now that weights are in BERT format"""
    model = Mymodel(model_name_or_path=model_path, args=args).to(device)

    # Load state dict with proper error handling
    state_dict_path = f"{model_path}/pytorch_model.bin"
    print(f"[INFO] Loading model weights from: {state_dict_path}")

    try:
        state_dict = torch.load(state_dict_path, map_location=device)

        # ✅ NO MORE MAPPING NEEDED - weights are already in BERT format
        print("[INFO] Loading BERT-format weights...")

        # Get current model state dict
        model_state_dict = model.state_dict()

        # Filter state dict to only include matching keys with correct shapes
        filtered_state_dict = {}
        missing_keys = []
        unexpected_keys = []
        shape_mismatch_keys = []

        for key, value in state_dict.items():
            if key in model_state_dict:
                if value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
                else:
                    shape_mismatch_keys.append(key)
                    print(f"[SHAPE MISMATCH] {key}: {value.shape} vs {model_state_dict[key].shape}")
            else:
                unexpected_keys.append(key)

        # Find missing keys
        for key in model_state_dict.keys():
            if key not in state_dict:
                missing_keys.append(key)

        # Load filtered state dict
        if filtered_state_dict:
            model.load_state_dict(filtered_state_dict, strict=False)
            loading_ratio = len(filtered_state_dict) / len(model_state_dict)
            print(
                f"[INFO] ✅ Successfully loaded {len(filtered_state_dict)}/{len(model_state_dict)} parameters ({loading_ratio:.1%})")

            if loading_ratio > 0.9:
                print("✅ Excellent! High parameter loading ratio.")
            elif loading_ratio > 0.7:
                print("⚠️ Acceptable parameter loading ratio.")
            else:
                print("❌ Low parameter loading ratio - model may not perform optimally.")
        else:
            print("[WARNING] ⚠️ No matching parameters found - using random initialization")

        # Print detailed debug information
        if missing_keys:
            print(
                f"[DEBUG] Missing keys ({len(missing_keys)}): {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
        if unexpected_keys:
            print(
                f"[DEBUG] Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
        if shape_mismatch_keys:
            print(
                f"[DEBUG] Shape mismatch keys ({len(shape_mismatch_keys)}): {shape_mismatch_keys[:5]}{'...' if len(shape_mismatch_keys) > 5 else ''}")

    except Exception as e:
        print(f"[ERROR] Failed to load model weights: {e}")
        print("[WARNING] Using randomly initialized model")

    model.eval()
    return model


def verify_model_embeddings(model, sample_texts, device):
    """Verify that model produces consistent embeddings"""
    print("\n[INFO] Verifying model embedding consistency...")

    with torch.no_grad():
        # Encode same text multiple times
        test_text = "what does jamaican people speak?"
        embeddings = []

        for i in range(3):
            emb = model.encode([test_text], convert_to_tensor=True)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

        # Check consistency
        cos_sim_1_2 = np.dot(embeddings[0][0], embeddings[1][0])
        cos_sim_1_3 = np.dot(embeddings[0][0], embeddings[2][0])

        print(f"[VERIFICATION] Cosine similarity between repeated encodings:")
        print(f"  Run 1 vs Run 2: {cos_sim_1_2:.6f}")
        print(f"  Run 1 vs Run 3: {cos_sim_1_3:.6f}")

        if cos_sim_1_2 > 0.999 and cos_sim_1_3 > 0.999:
            print("✅ Model produces consistent embeddings")
        else:
            print("⚠️ Model embeddings are inconsistent - possible initialization issues")

    return embeddings[0][0]


def evaluate_webq():
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

    # ✅ FIXED: Use improved model loading
    model = load_model_with_weights(MODEL_PATH, args, device)

    # ✅ NEW: Verify model consistency
    sample_embedding = verify_model_embeddings(model, ["test query"], device)

    # ✅ FIXED: Force rebuild cache to ensure embeddings match current model
    print("[INFO] Building corpus embeddings with current model state...")
    corpus_ids, corpus_embs = encode_corpus(model, corpus, force_rebuild=True)

    # Load the trained student weights safely (ignore missing/extra keys)
    state_dict = torch.load(f"{MODEL_PATH}/pytorch_model.bin", map_location=device)
    model.load_state_dict(state_dict, strict=False)

    model.eval()  # set to eval mode


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

                if num_eval == 0:  # only debug for first query
                    print("\n================= ADVANCED DEBUG (First Query) =================")
                    query_text = query_texts[j]
                    q_emb = batch_embs[j]

                    # --- Token-level analysis ---
                    from transformers import AutoTokenizer

                    # Initialize tokenizer once outside the loop (ideally at top of script)
                    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)

                    # Token-level analysis
                    q_tok = tokenizer(query_text, truncation=True, padding=True, return_tensors="pt")
                    query_tokens = tokenizer.convert_ids_to_tokens(q_tok["input_ids"][0].tolist())
                    print(f"Query Tokens: {query_tokens}")
                    print(f"Query norm: {np.linalg.norm(q_emb):.4f}")

                    # --- Cosine similarity with top & relevant docs ---
                    for rel_doc_id in relevant:
                        if rel_doc_id in corpus_ids:
                            rel_idx = corpus_ids.index(rel_doc_id)
                            rel_emb = corpus_embs_np[rel_idx]
                            print(f"Cos(query, {rel_doc_id}) = {np.dot(q_emb, rel_emb):.6f}")
                            print(f"Snippet: {corpus[rel_doc_id][:120]}...")
                        else:
                            print(f"[WARNING] Relevant doc {rel_doc_id} not found in corpus!")

                    # --- Top-K retrieval analysis ---
                    print("\nTop-10 retrieved docs (with CosSim):")
                    for rank, idx in enumerate(I[j], start=1):
                        doc_id = corpus_ids[idx]
                        sim = np.dot(q_emb, corpus_embs_np[idx])
                        print(f"Rank {rank:02d}: {doc_id}, CosSim={sim:.6f}, Snippet: {corpus[doc_id][:100]}")

                    # --- Detailed cached embedding inspection for first query ---
                    cache_file = f"corpus_embs_webq_{compute_hash(MODEL_PATH, CORPUS_FILE, len(corpus), BATCH_SIZE)}.pt"
                    meta_file = cache_file + ".meta.json"

                    if os.path.exists(cache_file):
                        cache = torch.load(cache_file, map_location="cpu")
                        ids, embs = cache["ids"], cache["embs"]
                        print(f"[CACHE DEBUG] Loaded {len(ids)} cached embeddings with shape {embs.shape}")

                        # Normalize cached embeddings
                        embs = embs / (embs.norm(dim=-1, keepdim=True) + 1e-12)
                        embs_np = embs.numpy().astype("float32")

                        # Check norms of some embeddings
                        sample_idx = 0
                        print(f"[CACHE DEBUG] Norm of first cached embedding: {torch.norm(embs[sample_idx]):.4f}")

                        # Compare query embedding with relevant docs in cache
                        for rel_doc_id in relevant:
                            if rel_doc_id in ids:
                                pos = ids.index(rel_doc_id)
                                cos_sim = np.dot(q_emb, embs_np[pos])
                                print(f"[CACHE DEBUG] Cos(query, cached[{rel_doc_id}]) = {cos_sim:.6f}")
                                print(f"[CACHE DEBUG] Snippet: {corpus[rel_doc_id][:120]}...")
                            else:
                                print(f"[CACHE DEBUG] Relevant doc {rel_doc_id} not found in cached embeddings")

                        # Optional: top-10 similarity with all cached embeddings
                        all_cos_sims = np.dot(embs_np, q_emb)
                        top10_indices = np.argsort(-all_cos_sims)[:10]
                        print("[CACHE DEBUG] Top-10 docs by cached similarity:")
                        for rank, idx in enumerate(top10_indices, start=1):
                            doc_id = ids[idx]
                            sim = all_cos_sims[idx]
                            snippet = corpus[doc_id][:100]
                            print(f"Rank {rank}: {doc_id}, CosSim={sim:.6f}, Snippet={snippet}...")

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
    output_excel = f"webq_eval_results_top{RECALL_K}.xlsx"
    df.to_excel(output_excel, index=False)
    print(f"[INFO] Saved detailed results to: {output_excel}")

if __name__ == "__main__":
    evaluate_webq()


