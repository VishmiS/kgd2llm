# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/evaluate/test_mmarco_EL.py

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
from langdetect import detect
from entity_linking.pipeline import enrich_query_with_entities_and_facts
import time

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="FEATURE_EXTRACTION"
)

# Paths - Updated to match your training data structure
BASE_MODEL_DIR = "/root/pycharm_semanticsearch/PATH_TO_OUTPUT_MODEL/mmarco"
DATA_DIR = "/root/pycharm_semanticsearch/dataset"
OUTPUTS_DIR = "/root/pycharm_semanticsearch"

# Checkpoints to evaluate
CHECKPOINTS = [f"checkpoint-epoch-5"]
RESULTS_FILE = os.path.join(OUTPUTS_DIR, "zevaluation_results_mmarco_EL.csv")
DETAILED_RESULTS_FILE = os.path.join(OUTPUTS_DIR, "zdetailed_evaluation_results_mmarco_EL.xlsx")

# Use the same data files as training
QUERIES_FILE = os.path.join(DATA_DIR, "ms_marco/test/queries.tsv")
CORPUS_FILE = os.path.join(DATA_DIR, "ms_marco/test/corpus.tsv")
QRELS_FILE = os.path.join(DATA_DIR, "ms_marco/test/qrels.tsv")

# Settings
MAX_QUERIES = 6000  # process all queries
MAX_CORPUS_DOCS = 10000000  # process all corpus documents
RECALL_K = 10
BATCH_SIZE = 16

# Entity Linking Cache Files
ENRICHED_QUERIES_CACHE = "enriched_queries_mmarco.pkl"
ENRICHED_RECORDS_CACHE = "enriched_records_mmarco.pkl"

args = Namespace(
    num_heads=8,
    ln=True,
    norm=True,
    padding_side='right',
    neg_K=3,
    max_seq_length=256,
    hidden_dim=768,
    output_dim=512,
    base_model_dir="bert-base-uncased"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

            query_id = str(row['id']).strip()
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

            corpus_id = str(row['id']).strip()
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
            passage_id = str(row['corpus_id']).strip()
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
    #
    # for qid, docs in qrels.items():
    #     if qid in queries:  # Query exists
    #         available_docs = docs & set(corpus.keys())  # Docs that exist in corpus
    #         available_relevant_pairs += len(available_docs)
    #
    # coverage = available_relevant_pairs / total_relevant_pairs if total_relevant_pairs > 0 else 0
    # print(f"Relevance judgment coverage: {available_relevant_pairs}/{total_relevant_pairs} ({coverage:.1%})")

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

    # Compute unique cache file name based on model and corpus info
    cache_hash = compute_hash(MODEL_PATH, CORPUS_FILE, len(corpus), batch_size)
    cache_file = f"corpus_embs_mmarco_{cache_hash}.pt"
    meta_file = f"{cache_file}.meta.json"

    # Try loading existing cache
    corpus_hash = hashlib.md5("\n".join(corpus_texts).encode("utf-8")).hexdigest()
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

    # Encode from scratch if cache doesn't exist or is invalid
    print("[INFO] Encoding corpus from scratch...")
    corpus_embs_list = []

    with torch.no_grad():
        for i in tqdm(range(0, len(corpus_texts), batch_size), desc="Encoding Corpus"):
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
            inputs = model.tokenizer(
                [test_text],
                padding='max_length',
                max_length=args.max_seq_length,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            emb = model.get_sentence_embedding(**inputs)
            emb = F.normalize(emb, p=2, dim=1)
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
    test_checkpoint = "checkpoint-epoch-5"  # Your best checkpoint
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
            inputs = model.tokenizer(
                sample_texts,
                padding='max_length',
                max_length=args.max_seq_length,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            sample_embs = model.get_sentence_embedding(**inputs)
            print(f"✅ Sample embeddings shape: {sample_embs.shape}")
            print(f"✅ Sample embeddings norm: {sample_embs.norm(dim=1)}")
    else:
        print(f"❌ Checkpoint missing: {model_path}")
        return False

    return data_ok


# ======== ENTITY LINKING AND REFORMULATION FUNCTIONS ========

def enrich_queries_with_entity_linking(queries, query_ids, force_rebuild=False):
    """Apply entity linking and query reformulation to all queries"""

    if force_rebuild or not (os.path.exists(ENRICHED_QUERIES_CACHE) and os.path.exists(ENRICHED_RECORDS_CACHE)):
        query_texts = []
        enriched_records = []
        print("[INFO] Starting entity linking enrichment and extraction...")

        for idx, qid in enumerate(tqdm(query_ids, desc="Entity Linking")):
            query_text_original = queries[qid].strip()
            query_lang = "unknown"

            try:
                query_lang = detect(query_text_original) if query_text_original else "unknown"
            except Exception:
                pass  # language detection can fail for short or non-text queries

            # ✅ Always enrich query regardless of language
            enriched_result = enrich_query_with_entities_and_facts(query_text_original)

            # Extract fact sets from both sources
            wikidata_facts_filtered = enriched_result.get("wikidata_facts_filtered", {})
            dbpedia_facts_filtered = enriched_result.get("dbpedia_facts_filtered", {})

            # Determine if any *additional facts* are actually found
            has_additional_facts = bool(wikidata_facts_filtered or dbpedia_facts_filtered)

            # Reformulate only if additional facts exist
            if has_additional_facts:
                # Use the final reformulated text from pipeline summary if present
                query_text = (
                        enriched_result.get("reformulated_query")
                        or enriched_result.get("natural_language_summary")
                        or query_text_original
                ).strip()
            else:
                query_text = query_text_original.strip()

            # Clean up whitespace and punctuation
            query_text = re.sub(r"[\s]+", " ", query_text)
            query_text = re.sub(r"^[\.\s]+", "", query_text)

            # Track whether reformulated
            reformulated_flag = "Yes" if query_text != query_text_original else "No"
            query_texts.append(query_text)

            all_entities = enriched_result.get("falcon_qids", {})
            wikidata_entities = enriched_result.get("wikidata_entities", {})
            wikidata_facts_filtered = enriched_result.get("wikidata_facts_filtered", {})
            dbpedia_entities = enriched_result.get("dbpedia_entities", {})
            dbpedia_facts_filtered = enriched_result.get("dbpedia_facts_filtered", {})

            # Entity complexity classification
            entity_complexity = {
                eid: ("simple" if len(str(e)) <= 20 else "complex") for eid, e in all_entities.items()
            }
            count_simple_entities = sum(v == "simple" for v in entity_complexity.values())
            count_complex_entities = sum(v == "complex" for v in entity_complexity.values())

            wikidata_facts_raw = (
                    enriched_result.get("wikidata_facts_filtered")
                    or enriched_result.get("wikidata_facts")
                    or {}
            )

            dbpedia_facts_raw = (
                    enriched_result.get("dbpedia_facts_filtered")
                    or enriched_result.get("dbpedia_facts_raw")
                    or {}
            )

            additional_info = []

            # --- Wikidata raw facts ---
            for facts in wikidata_facts_raw.values():
                for prop, val in facts:
                    additional_info.append(f"[Wikidata] {prop}: {val}")

            # --- DBpedia raw facts ---
            for facts in dbpedia_facts_raw.values():
                for prop, val in facts:
                    additional_info.append(f"[DBpedia] {prop}: {val}")

            enriched_records.append({
                "query_id": qid,
                "original_query": query_text_original,
                "reformulated_query": query_text,
                "was_reformulated": reformulated_flag,
                "query_language": query_lang,
                "query_complexity": (
                    "simple" if len(query_text_original.split()) <= 5 else
                    "medium" if len(query_text_original.split()) <= 12 else
                    "complex"
                ),
                "all_entities": list(all_entities.values()),
                "entity_qids": list(all_entities.keys()),
                "entity_complexity": entity_complexity,
                "count_entities": len(all_entities),
                "count_simple_entities": count_simple_entities,
                "count_complex_entities": count_complex_entities,
                "dbpedia_entities": dbpedia_entities,
                "entities_source": {
                    "wikidata_names": list(all_entities.values()),
                    "wikidata_qids": list(all_entities.keys()),
                    "dbpedia_names": list(dbpedia_entities.values()),
                    "dbpedia_urls": list(dbpedia_entities.keys())
                },
                "wikidata_facts_filtered": wikidata_facts_filtered,
                "wikidata_facts_raw": enriched_result.get("wikidata_facts", {}),
                "dbpedia_facts_filtered": dbpedia_facts_filtered,
                "dbpedia_facts_raw": enriched_result.get("dbpedia_facts_raw", {}),
                "additional_info": additional_info,

                "entity_types": enriched_result.get("entity_types", {}),
                "relation_entity_mapping": enriched_result.get("relation_entity_mapping", {}),
                "falcon_relations": enriched_result.get("falcon_relations", {}),
                "falcon_qids": enriched_result.get("falcon_qids", {}),

            })

            # 🧠 Periodically save partial progress to disk
            if (idx + 1) % 200 == 0:  # every 200 queries
                with open(ENRICHED_QUERIES_CACHE, "wb") as f:
                    pickle.dump(query_texts, f)
                with open(ENRICHED_RECORDS_CACHE, "wb") as f:
                    pickle.dump(enriched_records, f)
                print(f"[INFO] Saved progress: {idx + 1}/{len(query_ids)} queries processed...")

            time.sleep(3)  # if you want the pause

        # ✅ Final save once everything is done
        with open(ENRICHED_QUERIES_CACHE, "wb") as f:
            pickle.dump(query_texts, f)
        with open(ENRICHED_RECORDS_CACHE, "wb") as f:
            pickle.dump(enriched_records, f)
        print(f"[INFO] Finished and saved all enriched data to {ENRICHED_QUERIES_CACHE} and {ENRICHED_RECORDS_CACHE}")
    else:
        # Load from cache to skip enrichment step
        print("[INFO] Loading preprocessed entity linking cache...")
        with open(ENRICHED_QUERIES_CACHE, "rb") as f:
            query_texts = pickle.load(f)
        with open(ENRICHED_RECORDS_CACHE, "rb") as f:
            enriched_records = pickle.load(f)
        print(f"[INFO] Loaded {len(query_texts)} cached enriched queries.")

    return query_texts, enriched_records


def create_enrichment_dataframes(enriched_records):
    """Create DataFrames for queries and entities from enriched records"""
    query_rows = []
    entity_rows = []

    for rec in enriched_records:
        # Query-level info
        query_rows.append({
            "query_id": rec["query_id"],
            "original_query": rec["original_query"],
            "reformulated_query": rec["reformulated_query"],
            "was_reformulated": rec["was_reformulated"],
            "query_language": rec["query_language"],
            "query_complexity": rec["query_complexity"],
            "count_entities": rec["count_entities"],
            "count_simple_entities": rec["count_simple_entities"],
            "count_complex_entities": rec["count_complex_entities"],
        })

        enriched = rec.get("enriched", rec)  # alias for clarity

        # ---------------- RELATION + FACT SUMMARIES ----------------
        falcon_relations = enriched.get("falcon_relations", {})
        relation_mapping = enriched.get("relation_entity_mapping", {})
        falcon_qids = enriched.get("falcon_qids", {})

        # Identify relations by their *label*
        identified_relations = ", ".join(
            sorted(set(v for v in falcon_relations.values() if isinstance(v, str)))
        ) if falcon_relations else "(none)"

        # Build readable relation→entity summary
        relation_pairs = []
        for rel, ents in relation_mapping.items():
            rel_label = falcon_relations.get(rel, rel)
            if isinstance(ents, list):
                for e in ents:
                    if isinstance(e, dict):
                        src = e.get("source")
                        tgt = e.get("target") or e.get("value")
                        src_label = falcon_qids.get(src, src)
                        tgt_label = falcon_qids.get(tgt, tgt)
                        relation_pairs.append(f"{rel_label}: {src_label} → {tgt_label}")
                    else:
                        readable_ent = falcon_qids.get(e, e)
                        relation_pairs.append(f"{rel_label} → {readable_ent}")
            elif isinstance(ents, str):
                readable_ent = falcon_qids.get(ents, ents)
                relation_pairs.append(f"{rel_label} → {readable_ent}")
            elif isinstance(ents, dict):
                src = ents.get("source")
                tgt = ents.get("target") or ents.get("value")
                src_label = falcon_qids.get(src, src)
                tgt_label = falcon_qids.get(tgt, tgt)
                relation_pairs.append(f"{rel_label}: {src_label} → {tgt_label}")

        relation_mapping_summary = "; ".join(relation_pairs) if relation_pairs else "(none)"

        def summarize_facts(facts_dict):
            all_pairs = []
            for eid, facts in facts_dict.items():
                for f in facts:
                    if isinstance(f, dict):
                        all_pairs.append(f"{f.get('property')}: {f.get('value')}")
                    else:
                        try:
                            p, v = f
                            all_pairs.append(f"{p}: {v}")
                        except Exception:
                            continue
            return "; ".join(all_pairs[:5]) if all_pairs else "(none)"

        wikidata_facts_summary = summarize_facts(enriched.get("wikidata_facts_filtered", {}))
        dbpedia_facts_summary = summarize_facts(enriched.get("dbpedia_facts_filtered", {}))

        query_rows[-1].update({
            "identified_relations": identified_relations,
            "relation_mapping_summary": relation_mapping_summary,
            "wikidata_facts_summary": wikidata_facts_summary,
            "dbpedia_facts_summary": dbpedia_facts_summary,
        })

        # ---------------- ENTITY-LEVEL DETAILS ----------------
        all_entities = enriched.get("all_entities", [])
        entity_qids = enriched.get("entity_qids", [])
        complexity_map = enriched.get("entity_complexity", {})

        entity_types_dict = enriched.get("entity_types", {})
        relation_entity_mapping = enriched.get("relation_entity_mapping", {})

        # Try to get entity descriptions (from Wikidata labels)
        entity_descriptions_dict = {}
        if "wikidata_entities" in enriched:
            for qid, forms in enriched["wikidata_entities"].items():
                if isinstance(forms, list) and forms:
                    entity_descriptions_dict[qid] = forms[0]
                else:
                    entity_descriptions_dict[qid] = "(none)"

        for i, entity_name in enumerate(all_entities):
            entity_qid = entity_qids[i] if i < len(entity_qids) else None
            complexity = complexity_map.get(entity_qid, "unknown")

            # --- Wikidata facts ---
            wikidata_facts_filtered = enriched.get("wikidata_facts_filtered", {}).get(entity_qid, [])
            wikidata_facts_raw = enriched.get("wikidata_facts_raw", {}).get(entity_qid, [])

            def join_facts(facts):
                pairs = []
                for f in facts:
                    if isinstance(f, dict):
                        pairs.append(f"{f.get('property')}: {f.get('value')}")
                    else:
                        try:
                            p, v = f
                            pairs.append(f"{p}: {v}")
                        except Exception:
                            continue
                return "; ".join(pairs) if pairs else "(none)"

            wikidata_facts_filtered_str = join_facts(wikidata_facts_filtered)
            wikidata_facts_raw_str = join_facts(wikidata_facts_raw)

            # --- Entity types ---
            entity_types_list = entity_types_dict.get(entity_qid, [])
            entity_type_str = ", ".join(entity_types_list) if entity_types_list else "(unknown)"

            # --- Relations (relation names + linked entities) ---
            relations_for_entity = []

            falcon_relations = enriched.get("falcon_relations", {})
            falcon_qids = enriched.get("falcon_qids", {})
            relation_entity_mapping = enriched.get("relation_entity_mapping", {})

            for rel, ents in relation_entity_mapping.items():
                readable_rel = falcon_relations.get(rel, rel)  # e.g. P50 → write
                for e in ents:
                    if isinstance(e, dict):
                        # Handle modern relation structure
                        src = e.get("source")
                        tgt = e.get("target") or e.get("value")
                        src_label = falcon_qids.get(src, src)
                        tgt_label = falcon_qids.get(tgt, tgt)
                        if src == entity_qid or src_label.lower() == entity_name.lower():
                            relations_for_entity.append(f"{readable_rel}: {tgt_label}")
                        elif tgt == entity_qid or tgt_label.lower() == entity_name.lower():
                            relations_for_entity.append(f"{readable_rel}: {src_label}")
                    elif isinstance(e, str):
                        readable_ent = falcon_qids.get(e, e)
                        relations_for_entity.append(f"{readable_rel}: {readable_ent}")
                    else:
                        relations_for_entity.append(f"{readable_rel}: {str(e)}")

            relations_str = "; ".join(relations_for_entity) if relations_for_entity else "(none)"

            # For your identified_relations column:
            identified_relations = (
                ", ".join(sorted(set(falcon_relations.get(r, r) for r in relation_entity_mapping.keys())))
                if relation_entity_mapping else "(none)"
            )

            # --- URL ---
            entity_url = (
                f"https://www.wikidata.org/wiki/{entity_qid}"
                if entity_qid and entity_qid.startswith("Q")
                else "(unknown)"
            )

            # ---- Wikidata row ----
            entity_rows.append({
                "query_id": rec["query_id"],
                "entity_label": entity_name,
                "entity_qid": entity_qid,
                "entity_url": entity_url,
                "source": "Wikidata",
                "complexity": complexity,
                "entity_types": entity_type_str,
                "relations": relations_str,
                "raw_facts": wikidata_facts_raw_str,
                "filtered_facts": wikidata_facts_filtered_str
            })

            # ---- DBpedia row ----
            dbpedia_uri = None
            for uri, name in enriched.get("dbpedia_entities", {}).items():
                if name.lower() == entity_name.lower():
                    dbpedia_uri = uri
                    break

            dbpedia_facts_filtered = enriched.get("dbpedia_facts_filtered", {}).get(dbpedia_uri, [])
            dbpedia_facts_raw = enriched.get("dbpedia_facts_raw", {}).get(dbpedia_uri, [])

            dbpedia_facts_filtered_str = join_facts(dbpedia_facts_filtered)
            dbpedia_facts_raw_str = join_facts(dbpedia_facts_raw)

            entity_rows.append({
                "query_id": rec["query_id"],
                "entity_label": entity_name,
                "entity_qid": entity_qid,
                "entity_url": dbpedia_uri or "(unknown)",
                "source": "DBpedia",
                "complexity": complexity,
                "entity_types": "(unknown)",
                "relations": relations_str,
                "raw_facts": dbpedia_facts_raw_str,
                "filtered_facts": dbpedia_facts_filtered_str
            })

    return query_rows, entity_rows


def get_correct_answer(corpus, relevant_doc_ids):
    """Extract correct answers from relevant documents"""
    correct_answers = []
    for doc_id in relevant_doc_ids:
        if doc_id in corpus:
            doc_text = corpus[doc_id]
            # Try to extract the answer part (after the query)
            if '?' in doc_text:
                answer_part = doc_text.split('?', 1)[-1].strip()
                if answer_part:
                    correct_answers.append(answer_part)
            else:
                correct_answers.append(doc_text)

    return correct_answers if correct_answers else ["Answer not found in corpus"]


def generate_human_readable_explanation(ranked_doc_ids, relevant_doc_ids, top_docs_sample):
    """Generate human-readable explanation for retrieval performance"""
    first_relevant_rank = None
    for rank, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id in relevant_doc_ids:
            first_relevant_rank = rank
            break

    if first_relevant_rank is None:
        return "No relevant documents found in top 10 results."
    elif first_relevant_rank == 1:
        return "First relevant document found at rank 1."
    else:
        return f"First relevant document found at rank {first_relevant_rank}."


def get_top_docs_sample(corpus, ranked_doc_ids, max_samples=3):
    """Get sample of top retrieved documents with their content"""
    samples = []
    for doc_id in ranked_doc_ids[:max_samples]:
        if doc_id in corpus:
            samples.append((doc_id, corpus[doc_id][:100] + "..." if len(corpus[doc_id]) > 100 else corpus[doc_id]))
        else:
            samples.append((doc_id, "Document not found in corpus"))
    return samples


def evaluate_mmarco():
    """Main evaluation function for MS MARCO with Entity Linking"""
    global MODEL_PATH

    print("\n" + "=" * 60)
    print("MS MARCO EVALUATION WITH ENTITY LINKING")
    print("=" * 60)

    # Verify critical files exist
    critical_files = [QUERIES_FILE, CORPUS_FILE, QRELS_FILE, MODEL_PATH]
    for f in critical_files:
        if not os.path.exists(f):
            print(f"❌ Critical file missing: {f}")
            return 0.0, 0.0, []

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
        return 0.0, 0.0, []

    # Filter to queries that have relevance judgments AND exist in our queries
    valid_queries = {qid: qtext for qid, qtext in queries.items()
                     if qid in qrels and qrels[qid]}

    print(f"✅ Using {len(valid_queries)} valid queries for evaluation")

    if len(valid_queries) == 0:
        print("❌ No valid queries for evaluation!")
        return 0.0, 0.0, []

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

    # ======== ENTITY LINKING AND QUERY REFORMULATION ========
    print("\n" + "=" * 60)
    print("[ENTITY LINKING] Applying query enrichment and reformulation")
    print("=" * 60)

    query_ids = list(valid_queries.keys())
    query_texts, enriched_records = enrich_queries_with_entity_linking(valid_queries, query_ids, force_rebuild=False)

    # Create enrichment dataframes for analysis
    query_rows, entity_rows = create_enrichment_dataframes(enriched_records)

    mrr_total, recall_total, num_eval = 0, 0, 0
    detailed_results = []

    print(f"\n[INFO] Evaluating {len(query_ids)} queries...")

    with torch.no_grad():
        for i in tqdm(range(0, len(query_texts), BATCH_SIZE), desc="Evaluating Queries"):
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
                relevant_doc_ids = list(qrels.get(qid, set()))

                # Compute MRR
                reciprocal_rank = 0
                for rank, doc_id in enumerate(ranked_doc_ids, start=1):
                    if doc_id in relevant_doc_ids:
                        reciprocal_rank = 1.0 / rank
                        break
                mrr_total += reciprocal_rank

                # Compute Recall@K
                recall_at_k = 1 if set(relevant_doc_ids) & set(ranked_doc_ids) else 0
                recall_total += recall_at_k

                # Find enrichment record for this query
                enrichment_record = next((rec for rec in enriched_records if rec["query_id"] == qid), {})

                # Generate detailed results
                correct_answers = get_correct_answer(corpus, relevant_doc_ids)
                explanation = generate_human_readable_explanation(ranked_doc_ids, relevant_doc_ids, [])
                top_docs_sample = get_top_docs_sample(corpus, ranked_doc_ids)

                # Calculate similarity scores
                query_emb = batch_embs[j]
                top_doc_similarity = D[j][0] if len(D[j]) > 0 else 0.0

                # Find highest similarity among relevant documents
                relevant_similarity = 0.0
                relevant_rank = None

                for rank, (doc_id, sim_score) in enumerate(zip(ranked_doc_ids, D[j]), start=1):
                    if doc_id in relevant_doc_ids:
                        relevant_similarity = max(relevant_similarity, sim_score)
                        if relevant_rank is None:
                            relevant_rank = rank

                # If no relevant document found in top-K, check all relevant documents
                if relevant_similarity == 0.0 and relevant_doc_ids:
                    for rel_doc_id in relevant_doc_ids:
                        if rel_doc_id in corpus_ids:
                            rel_idx = corpus_ids.index(rel_doc_id)
                            rel_sim = np.dot(query_emb, corpus_embs_np[rel_idx])
                            relevant_similarity = max(relevant_similarity, rel_sim)

                detailed_results.append({
                    'query_id': qid,
                    'query_text': valid_queries[qid],
                    'reformulated_query': enrichment_record.get("reformulated_query", valid_queries[qid]),
                    'was_reformulated': enrichment_record.get("was_reformulated", "No"),
                    'relevant_doc_ids': relevant_doc_ids,
                    'top_10_doc_ids': ranked_doc_ids,
                    'MRR': reciprocal_rank,
                    'Recall@10': recall_at_k,
                    'correct_answers': correct_answers,
                    'Human_readable_explanation': explanation,
                    'Top_docs_sample': top_docs_sample,
                    'relevant_doc_similarity': relevant_similarity,
                    'top_doc_similarity': top_doc_similarity,
                    'relevant_doc_rank': relevant_rank if relevant_rank is not None else "Not in top-K",
                    'entity_count': enrichment_record.get("count_entities", 0),
                    'reformulation_impact': "Yes" if enrichment_record.get("was_reformulated") == "Yes" else "No"
                })

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

    # Calculate reformulation impact
    reformulated_results = [r for r in detailed_results if r['reformulation_impact'] == 'Yes']
    original_results = [r for r in detailed_results if r['reformulation_impact'] == 'No']

    if len(reformulated_results) > 0:
        reformulated_mrr = np.mean([r['MRR'] for r in reformulated_results])
        reformulated_recall = np.mean([r['Recall@10'] for r in reformulated_results])
        print(f"\n[REFORMULATION ANALYSIS]")
        print(f"Reformulated queries: {len(reformulated_results)}")
        print(f"  MRR@{RECALL_K}    : {reformulated_mrr:.4f}")
        print(f"  Recall@{RECALL_K} : {reformulated_recall:.4f}")

    if len(original_results) > 0:
        original_mrr = np.mean([r['MRR'] for r in original_results])
        original_recall = np.mean([r['Recall@10'] for r in original_results])
        print(f"Original queries: {len(original_results)}")
        print(f"  MRR@{RECALL_K}    : {original_mrr:.4f}")
        print(f"  Recall@{RECALL_K} : {original_recall:.4f}")

    return avg_mrr, avg_recall, detailed_results, query_rows, entity_rows


def save_detailed_results_to_excel(detailed_results, query_rows, entity_rows, checkpoint_name):
    """Save detailed results to Excel file"""
    if not detailed_results:
        print("❌ No detailed results to save")
        return

    # Convert to DataFrames
    df_data = []
    for result in detailed_results:
        row = {
            'query_id': result['query_id'],
            'query_text': result['query_text'],
            'reformulated_query': result['reformulated_query'],
            'was_reformulated': result['was_reformulated'],
            'relevant_doc_ids': str(result['relevant_doc_ids']),
            'top_10_doc_ids': str(result['top_10_doc_ids']),
            'MRR': result['MRR'],
            'Recall@10': result['Recall@10'],
            'correct_answers': ' | '.join(result['correct_answers']),
            'Human_readable_explanation': result['Human_readable_explanation'],
            'Top_docs_sample': str(result['Top_docs_sample']),
            'relevant_doc_similarity': result['relevant_doc_similarity'],
            'top_doc_similarity': result['top_doc_similarity'],
            'relevant_doc_rank': result['relevant_doc_rank'],
            'entity_count': result['entity_count'],
            'reformulation_impact': result['reformulation_impact']
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)
    queries_df = pd.DataFrame(query_rows)
    entities_df = pd.DataFrame(entity_rows)

    # Save to Excel
    try:
        with pd.ExcelWriter(DETAILED_RESULTS_FILE, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=f'Evaluation_Results', index=False)
            queries_df.to_excel(writer, sheet_name='Enriched_Queries', index=False)
            entities_df.to_excel(writer, sheet_name='Entity_Details', index=False)

            # Add summary sheet
            summary_data = {
                'Metric': ['Total Queries', 'Average MRR@10', 'Average Recall@10', 'Checkpoint',
                           'Reformulated Queries', 'Original Queries'],
                'Value': [len(df), df['MRR'].mean(), df['Recall@10'].mean(), checkpoint_name,
                          len([r for r in detailed_results if r['reformulation_impact'] == 'Yes']),
                          len([r for r in detailed_results if r['reformulation_impact'] == 'No'])]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

        print(f"✅ Detailed results saved to: {DETAILED_RESULTS_FILE}")

        # Print first few rows for verification
        print("\n📊 First 3 rows of detailed results:")
        print(df.head(3).to_string(index=False))

    except Exception as e:
        print(f"❌ Failed to save Excel file: {e}")
        # Fallback to CSV
        csv_file = DETAILED_RESULTS_FILE.replace('.xlsx', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"✅ Results saved to CSV as fallback: {csv_file}")


def evaluate_all_checkpoints():
    """Evaluate all checkpoints and save results to files"""
    # Create results file with header
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ['checkpoint', 'epoch', 'mrr@10', 'recall@10', 'num_queries', 'reformulated_queries', 'timestamp'])

    print(f"Evaluating {len(CHECKPOINTS)} checkpoints...")
    print(f"Results will be saved to: {RESULTS_FILE}")
    print(f"Detailed results will be saved to: {DETAILED_RESULTS_FILE}")

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
            mrr, recall, detailed_results, query_rows, entity_rows = evaluate_mmarco()

            # Count reformulated queries
            reformulated_count = len([r for r in detailed_results if r.get('reformulation_impact') == 'Yes'])

            # Save summary results to CSV
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [checkpoint, epoch_num, mrr, recall, len(detailed_results), reformulated_count, datetime.now()])

            # Save detailed results to Excel
            save_detailed_results_to_excel(detailed_results, query_rows, entity_rows, checkpoint)

            print(f"✅ Completed: {checkpoint} - MRR@10: {mrr:.4f}, Recall@10: {recall:.4f}")
            print(f"   Reformulated queries: {reformulated_count}/{len(detailed_results)}")

        except Exception as e:
            print(f"❌ Error evaluating {checkpoint}: {e}")
            import traceback
            traceback.print_exc()
            # Save error result
            with open(RESULTS_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([checkpoint, epoch_num, 0.0, 0.0, 0, 0, datetime.now(), f"Error: {str(e)}"])

    print(f"\n{'=' * 80}")
    print("Evaluation completed!")
    print(f"Summary results saved to: {RESULTS_FILE}")
    print(f"Detailed results saved to: {DETAILED_RESULTS_FILE}")

    # Print summary
    try:
        results_df = pd.read_csv(RESULTS_FILE)
        print("\nSummary of results:")
        print(results_df[['checkpoint', 'epoch', 'mrr@10', 'recall@10', 'reformulated_queries']].to_string(index=False))

        # Find best checkpoint
        if len(results_df) > 0:
            best_idx = results_df['mrr@10'].idxmax()
            best_row = results_df.loc[best_idx]
            print(f"\n🏆 BEST CHECKPOINT: {best_row['checkpoint']} with MRR@10: {best_row['mrr@10']:.4f}")

    except Exception as e:
        print(f"Could not read results file: {e}")


if __name__ == "__main__":
    # Run diagnostic first to identify issues
    print("🚀 Starting MS MARCO Evaluation with Entity Linking")
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
        print("   2. Verify the model checkpoints exist in: /root/pycharm_semanticsearch/PATH_TO_OUTPUT_MODEL/mmarco/")
        print("   3. Make sure your data files match the expected format shown in the diagnostic")