# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/evaluate/test_webq2.py

import sys, os
import torch.nn.functional as F
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
from transformers import AutoTokenizer
import csv
import pickle
from utils.common_utils import load_pickle
import json
from datetime import datetime

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)

# Paths - Exact paths from your data inspection
BASE_MODEL_DIR = "/root/pycharm_semanticsearch/PATH_TO_OUTPUT_MODEL/webq"
DATA_DIR = "/root/pycharm_semanticsearch/dataset"
OUTPUTS_DIR = "/root/pycharm_semanticsearch/outputs"

# Checkpoints to evaluate (epochs 1 through 8)
CHECKPOINTS = [f"checkpoint-epoch-{i}" for i in range(1, 16)]
RESULTS_FILE = os.path.join(OUTPUTS_DIR, "webq_evaluation_results.csv")

# EXACT PATHS FROM YOUR DATA INSPECTION
QUERIES_FILE = os.path.join(DATA_DIR, "web_questions/test/queries.tsv")
CORPUS_FILE = os.path.join(DATA_DIR, "web_questions/test/corpus.tsv")
QRELS_FILE = os.path.join(DATA_DIR, "web_questions/test/qrels.tsv")

# Settings
MAX_QUERIES = 3000  # process all queries
MAX_CORPUS_DOCS = 10000000  # process all corpus documents
RECALL_K = 10
BATCH_SIZE = 16
CORPUS_EMB_FILE = "corpus_embs_webq.pt"

args = Namespace(
    num_heads=8,
    ln=True,
    norm=True,
    padding_side='right',
    neg_K=3,
    max_seq_length=256,
    hidden_dim=768,           # Add this - from your model
    output_dim=512,           # Add this - from your model
    base_model_dir="bert-base-uncased"  # Add this
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable for model path
MODEL_PATH = ""


def load_queries(max_queries=None):
    """Load queries from TSV with exact format from inspection"""
    queries = {}
    try:
        df = pd.read_csv(QUERIES_FILE, sep='\t')
        print(f"✅ Loaded queries DataFrame with {len(df)} rows")

        for _, row in df.iterrows():
            if max_queries and len(queries) >= max_queries:
                break

            query_id = str(row['query_id']).strip()
            query_text = str(row['query']).strip()

            if query_id and query_text:
                queries[query_id] = query_text

        print(f"✅ Loaded {len(queries)} queries from {QUERIES_FILE}")

        # Print sample
        if queries:
            sample_id = list(queries.keys())[0]
            print(f"   Sample: ID='{sample_id}', Query='{queries[sample_id]}'")

    except Exception as e:
        print(f"❌ Failed to load queries: {e}")
        import traceback
        traceback.print_exc()

    return queries


def load_corpus(max_docs=None):
    """Load corpus from TSV with exact format from inspection"""
    corpus = {}
    try:
        df = pd.read_csv(CORPUS_FILE, sep='\t')
        print(f"✅ Loaded corpus DataFrame with {len(df)} rows")

        for _, row in df.iterrows():
            if max_docs and len(corpus) >= max_docs:
                break

            corpus_id = str(row['corpus_id']).strip()
            text = str(row['text']).strip()

            if corpus_id and text:
                corpus[corpus_id] = text

        print(f"✅ Loaded {len(corpus)} documents from {CORPUS_FILE}")

        # Print sample
        if corpus:
            sample_id = list(corpus.keys())[0]
            print(f"   Sample: ID='{sample_id}', Text='{corpus[sample_id][:100]}...'")

    except Exception as e:
        print(f"❌ Failed to load corpus: {e}")
        import traceback
        traceback.print_exc()

    return corpus


def load_qrels():
    """Load qrels from TSV with exact format from inspection"""
    qrels = defaultdict(set)
    try:
        df = pd.read_csv(QRELS_FILE, sep='\t')
        print(f"✅ Loaded qrels DataFrame with {len(df)} rows")

        for _, row in df.iterrows():
            query_id = str(row['query_id']).strip()
            passage_id = str(row['passage_id']).strip()
            rel = str(row['rel']).strip()

            if query_id and passage_id and rel == '1':
                qrels[query_id].add(passage_id)

        print(f"✅ Loaded {len(qrels)} query-doc relationships")
        print(f"   Total relevant pairs: {sum(len(docs) for docs in qrels.values())}")

        # Print sample
        if qrels:
            sample_qid = list(qrels.keys())[0]
            print(f"   Sample: Query='{sample_qid}', Relevant docs: {list(qrels[sample_qid])[:3]}")

    except Exception as e:
        print(f"❌ Failed to load qrels: {e}")
        import traceback
        traceback.print_exc()

    return qrels


def verify_data_consistency(queries, corpus, qrels):
    """Verify that data is loaded correctly with exact format"""
    print("\n" + "=" * 50)
    print("DATA CONSISTENCY CHECK")
    print("=" * 50)

    # Basic counts
    print(f"Total queries: {len(queries)}")
    print(f"Total corpus documents: {len(corpus)}")
    print(f"Total queries with relevance judgments: {len(qrels)}")

    # Check overlap between queries and qrels
    overlapping_queries = set(queries.keys()) & set(qrels.keys())
    print(f"Queries with both text and relevance judgments: {len(overlapping_queries)}")

    # Verify relevant documents exist in corpus
    all_relevant_docs = set()
    for docs in qrels.values():
        all_relevant_docs.update(docs)

    missing_docs = all_relevant_docs - set(corpus.keys())
    print(f"Relevant documents missing from corpus: {len(missing_docs)}")

    if missing_docs:
        print(f"First 3 missing docs: {list(missing_docs)[:3]}")

    # Calculate coverage statistics
    total_relevant_pairs = sum(len(docs) for docs in qrels.values())
    available_relevant_pairs = 0

    for qid, docs in qrels.items():
        if qid in queries:  # Query exists
            available_docs = docs & set(corpus.keys())  # Docs that exist in corpus
            available_relevant_pairs += len(available_docs)

    coverage = available_relevant_pairs / total_relevant_pairs if total_relevant_pairs > 0 else 0
    print(f"Relevance judgment coverage: {available_relevant_pairs}/{total_relevant_pairs} ({coverage:.1%})")

    # Check if we have enough data for meaningful evaluation
    if len(overlapping_queries) == 0:
        print("❌ CRITICAL: No overlapping queries between queries and qrels!")
        return False

    if len(overlapping_queries) < 10:
        print(f"⚠️ WARNING: Only {len(overlapping_queries)} queries available for evaluation")
        print("Proceeding anyway...")
        return True

    print("✅ Good data coverage for evaluation")
    return True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_hash(model_path, corpus_file, max_docs, batch_size):
    """Generate a unique cache hash based on key parameters including model weights."""
    model_file = f"{model_path}/pytorch_model.bin"
    model_info = ""
    if os.path.exists(model_file):
        stat = os.stat(model_file)
        model_info = f"{stat.st_mtime}-{stat.st_size}"

    info = f"{model_path}-{model_info}-{corpus_file}-{max_docs}-{batch_size}"
    return hashlib.md5(info.encode()).hexdigest()


def encode_corpus(model, corpus, batch_size=BATCH_SIZE, force_rebuild=False):
    """Use the SAME embedding method as training"""
    corpus_ids = list(corpus.keys())
    corpus_texts = [corpus[cid] for cid in corpus_ids]
    corpus_embs_list = []

    with torch.no_grad():
        for i in range(0, len(corpus_texts), batch_size):
            batch_texts = corpus_texts[i:i + batch_size]

            # Use the SAME tokenization as training
            inputs = model.tokenizer(
                batch_texts,
                padding='max_length',
                max_length=args.max_seq_length,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            # Use the SAME embedding method as training
            batch_embs = model.get_sentence_embedding(**inputs)
            batch_embs = F.normalize(batch_embs, p=2, dim=1)  # Normalize like training
            corpus_embs_list.append(batch_embs)

    corpus_embs = torch.cat(corpus_embs_list, dim=0)
    return corpus_ids, corpus_embs


def verify_model_functionality(model, device):
    """Verify model works the same as during training"""
    print("\n🔍 MODEL FUNCTIONALITY VERIFICATION")

    # Test the same way as training
    test_texts = ["what is the capital of france", "test query"]

    with torch.no_grad():
        # Method 1: Training-style embedding
        inputs = model.tokenizer(
            test_texts,
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt'
        ).to(device)

        train_style_embs = model.get_sentence_embedding(**inputs)

        print(f"✅ Training-style embeddings shape: {train_style_embs.shape}")
        print(f"✅ Training-style embeddings norm: {train_style_embs.norm(dim=1)}")

        # Check if embeddings are reasonable
        similarity = F.cosine_similarity(train_style_embs[0:1], train_style_embs[1:2])
        print(f"✅ Similarity between test queries: {similarity.item():.4f}")

        if similarity.item() > 0.95:
            print("⚠️  WARNING: Embeddings might be collapsing")

    return train_style_embs

def load_model_with_weights(model_path, args, device):
    """Load model with EXACT same architecture as training"""
    print(f"[INFO] Loading model with custom architecture from: {model_path}")

    # Load with same architecture as training
    model = Mymodel(
        model_name_or_path=model_path,
        args=args
    ).to(device)

    # Load state dict
    state_dict_path = f"{model_path}/pytorch_model.bin"
    print(f"[INFO] Loading weights from: {state_dict_path}")

    try:
        state_dict = torch.load(state_dict_path, map_location=device, weights_only=False)

        # Handle DeepSpeed wrapping
        if any(key.startswith('module.') for key in state_dict.keys()):
            print("[INFO] Removing 'module.' prefix from DeepSpeed state dict")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        # Load with strict=False to handle architecture differences
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        print(f"[INFO] ✅ Model loaded successfully")
        print(f"[INFO] Missing keys: {len(missing_keys)}")
        print(f"[INFO] Unexpected keys: {len(unexpected_keys)}")

        if missing_keys:
            print(f"[DEBUG] First 5 missing keys: {missing_keys[:5]}")
        if unexpected_keys:
            print(f"[DEBUG] First 5 unexpected keys: {unexpected_keys[:5]}")

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


def quick_diagnostic():
    """Run a quick diagnostic to identify the issue"""
    print("🔍 RUNNING DIAGNOSTIC...")

    # Test data loading
    queries = load_queries(10)  # Just load 10 for testing
    corpus = load_corpus(100)  # Just load 100 for testing
    qrels = load_qrels()

    print(f"Queries loaded: {len(queries)}")
    print(f"Corpus loaded: {len(corpus)}")
    print(f"Qrels loaded: {len(qrels)}")

    # Test data consistency
    data_ok = verify_data_consistency(queries, corpus, qrels)

    # Test model loading
    test_checkpoint = "checkpoint-epoch-3"  # Your best checkpoint
    model_path = os.path.join(BASE_MODEL_DIR, test_checkpoint)

    if os.path.exists(model_path):
        print(f"✅ Checkpoint exists: {model_path}")
        # Test a simple embedding
        model = load_model_with_weights(model_path, args, device)
        test_embedding = verify_model_embeddings(model, ["test query"], device)
        print(f"✅ Model produces embeddings of shape: {test_embedding.shape}")

        # Test encoding a sample
        sample_texts = ["test query 1", "test query 2"]
        with torch.no_grad():
            sample_embs = model.encode(sample_texts, convert_to_tensor=True)
            print(f"✅ Sample embeddings shape: {sample_embs.shape}")
            print(f"✅ Sample embeddings norm: {sample_embs.norm(dim=1)}")
    else:
        print(f"❌ Checkpoint missing: {model_path}")
        return False

    return data_ok


def evaluate_webq():
    """Main evaluation function for WebQuestions"""
    global MODEL_PATH

    print("\n" + "=" * 60)
    print("WEBQUESTIONS EVALUATION")
    print("=" * 60)

    # Verify critical files exist
    critical_files = [QUERIES_FILE, CORPUS_FILE, QRELS_FILE, MODEL_PATH]
    for f in critical_files:
        if not os.path.exists(f):
            print(f"❌ Critical file missing: {f}")
            return 0.0, 0.0

    set_seed(42)
    print("[INFO] Random seed fixed to 42 for reproducibility.")

    # Load data
    print("\n[INFO] Loading evaluation data...")
    queries = load_queries(MAX_QUERIES)
    corpus = load_corpus(MAX_CORPUS_DOCS)
    qrels = load_qrels()

    # Verify data consistency
    if not verify_data_consistency(queries, corpus, qrels):
        print("❌ CRITICAL: Data consistency check failed!")
        return 0.0, 0.0

    # Filter to queries that have relevance judgments AND exist in our queries
    valid_queries = {qid: qtext for qid, qtext in queries.items()
                     if qid in qrels and qrels[qid]}

    print(f"✅ Using {len(valid_queries)} valid queries for evaluation")

    if len(valid_queries) == 0:
        print("❌ No valid queries for evaluation!")
        return 0.0, 0.0

    # Load model
    print(f"\n[INFO] Loading model from: {MODEL_PATH}")
    model = load_model_with_weights(MODEL_PATH, args, device)

    # 🔥 CRITICAL: Verify model functionality
    verify_model_functionality(model, device)

    # Build corpus embeddings
    print("\n[INFO] Building corpus embeddings...")
    corpus_ids, corpus_embs = encode_corpus(model, corpus, force_rebuild=False)

    model.eval()

    # Check embedding quality
    corpus_embs_np = corpus_embs.cpu().numpy().astype('float32')
    print(f"[CORPUS EMBEDDINGS] Shape: {corpus_embs_np.shape}")
    print(f"[CORPUS EMBEDDINGS] Mean norm: {np.mean(np.linalg.norm(corpus_embs_np, axis=1)):.4f}")

    # FAISS CPU index (IP = inner product) with L2-normalized embeddings
    index_flat = faiss.IndexFlatIP(corpus_embs_np.shape[1])
    index_flat.add(corpus_embs_np)
    print(f"✅ FAISS index built with {index_flat.ntotal} documents")

    mrr_total, recall_total, num_eval = 0, 0, 0

    query_ids = list(valid_queries.keys())
    query_texts = [
        re.sub(r"^[\.\s]+", "", re.sub(r"[\s]+", " ", valid_queries[qid].strip()))
        for qid in query_ids
    ]

    print(f"\n[INFO] Evaluating {len(query_ids)} queries...")

    with torch.no_grad():
        printed_examples = 0
        for i in range(0, len(query_texts), BATCH_SIZE):
            batch_ids = query_ids[i:i + BATCH_SIZE]
            batch_texts = query_texts[i:i + BATCH_SIZE]

            # 🔥 FIXED: Use the SAME encoding method as corpus (NO template)
            inputs = model.tokenizer(
                batch_texts,
                padding='max_length',
                max_length=args.max_seq_length,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            # Use get_sentence_embedding directly (same as corpus encoding)
            batch_embs = model.get_sentence_embedding(**inputs)
            batch_embs = F.normalize(batch_embs, p=2, dim=1)  # Normalize like training
            batch_embs = batch_embs.cpu().numpy().astype("float32")

            # FAISS CPU search
            D, I = index_flat.search(batch_embs, RECALL_K)

            # Check for embedding collapse
            if i == 0:
                avg_similarity = np.mean(D)
                print(f"[DEBUG] Average top similarity score: {avg_similarity:.4f}")
                if avg_similarity > 0.99:
                    print("⚠️  WARNING: High similarity scores detected")

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
                if not relevant_doc_ids:
                    recall_at_k = 0
                else:
                    retrieved_relevant = len(set(relevant_doc_ids) & set(ranked_doc_ids))
                    recall_at_k = retrieved_relevant / len(relevant_doc_ids)
                recall_total += recall_at_k

                # Print first few examples for debugging
                if printed_examples < 2 and reciprocal_rank > 0:
                    print("\n--- Example (Successful Retrieval) ---")
                    print(f"Query   : {valid_queries[qid]}")
                    print(f"Relevant docs: {list(relevant)}")
                    print(f"Retrieved docs: {ranked_doc_ids[:5]}")
                    print(f"Similarity scores: {D[j][:5]}")
                    print(f"MRR     : {reciprocal_rank:.4f}")
                    printed_examples += 1

                num_eval += 1

    avg_mrr = mrr_total / num_eval if num_eval > 0 else 0
    avg_recall = recall_total / num_eval if num_eval > 0 else 0

    # Final results
    print("\n" + "=" * 60)
    print("[FINAL RESULTS]")
    print("=" * 60)
    print(f"Evaluated on {num_eval} queries")
    print(f"MRR@{RECALL_K}    : {avg_mrr:.4f}")
    print(f"Recall@{RECALL_K} : {avg_recall:.4f}")

    return avg_mrr, avg_recall


def evaluate_all_checkpoints():
    """Evaluate all checkpoints and save results to file"""
    # Create results file with header
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['checkpoint', 'epoch', 'mrr@10', 'recall@10', 'num_queries', 'timestamp'])

    print(f"Evaluating {len(CHECKPOINTS)} checkpoints...")
    print(f"Results will be saved to: {RESULTS_FILE}")

    for checkpoint in CHECKPOINTS:
        global MODEL_PATH
        MODEL_PATH = os.path.join(BASE_MODEL_DIR, checkpoint)

        if not os.path.exists(MODEL_PATH):
            print(f"❌ Checkpoint not found: {MODEL_PATH}")
            continue

        print(f"\n{'=' * 80}")
        print(f"Evaluating checkpoint: {checkpoint}")
        print(f"{'=' * 80}")

        try:
            # Extract epoch number from checkpoint name
            epoch_num = int(checkpoint.split('-')[-1])

            # Run evaluation
            mrr, recall = evaluate_webq()

            # Save results
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([checkpoint, epoch_num, mrr, recall, MAX_QUERIES, datetime.now()])

            print(f"✅ Completed: {checkpoint} - MRR@10: {mrr:.4f}, Recall@10: {recall:.4f}")

        except Exception as e:
            print(f"❌ Error evaluating {checkpoint}: {e}")
            import traceback
            traceback.print_exc()
            # Save error result
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([checkpoint, epoch_num, 0.0, 0.0, 0, datetime.now(), f"Error: {str(e)}"])

    print(f"\n{'=' * 80}")
    print("Evaluation completed! Results saved to:", RESULTS_FILE)

    # Print summary
    try:
        results_df = pd.read_csv(RESULTS_FILE)
        print("\nSummary of results:")
        print(results_df[['checkpoint', 'epoch', 'mrr@10', 'recall@10']].to_string(index=False))

        # Find best checkpoint
        if len(results_df) > 0:
            best_idx = results_df['mrr@10'].idxmax()
            best_row = results_df.loc[best_idx]
            print(f"\n🏆 BEST CHECKPOINT: {best_row['checkpoint']} with MRR@10: {best_row['mrr@10']:.4f}")

    except Exception as e:
        print(f"Could not read results file: {e}")


if __name__ == "__main__":
    # Run diagnostic first to identify issues
    print("🚀 Starting WebQuestions Evaluation")
    diagnostic_ok = quick_diagnostic()

    if diagnostic_ok:
        print("\n" + "=" * 80)
        print("DIAGNOSTIC PASSED - STARTING FULL EVALUATION")
        print("=" * 80)
        evaluate_all_checkpoints()
    else:
        print("\n❌ DIAGNOSTIC FAILED - Please fix the issues above before running full evaluation")
        print("\n💡 TROUBLESHOOTING:")
        print("   1. Check if all data files exist at the specified paths")
        print("   2. Verify the model checkpoints exist in: /root/pycharm_semanticsearch/PATH_TO_OUTPUT_MODEL/webq/")
        print("   3. Make sure your data files match the expected format shown in the diagnostic")