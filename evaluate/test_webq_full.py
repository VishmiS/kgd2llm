# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/evaluate/test_webq_full.py


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
from langdetect import detect
from entity_linking.pipeline import enrich_query_with_entities_and_facts, print_clean_pipeline_result
import pickle
import time

# Paths
MODEL_PATH = "/root/pycharm_semanticsearch/PATH_TO_OUTPUT_MODEL/webq/final_student_model_fp32"
QUERIES_FILE = "/root/pycharm_semanticsearch/dataset/web_questions/test/queries.tsv"
CORPUS_FILE  = "/root/pycharm_semanticsearch/dataset/web_questions/test/corpus.tsv"
QRELS_FILE   = "/root/pycharm_semanticsearch/dataset/web_questions/test/qrels.tsv"

# Settings
MAX_QUERIES = 100 # 101093       process all queries
MAX_CORPUS_DOCS = 1008986 # 1008986     limit corpus documents for testing
RECALL_K = 10
BATCH_SIZE = 16
CORPUS_EMB_FILE = "corpus_embs_webq_EL.pt"

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
    cache_file = f"corpus_embs_webq_EL_{cache_hash}.pt"

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

    # Normalize before saving for FAISS consistency
    corpus_embs_np = corpus_embs.cpu().numpy().astype("float32")
    faiss.normalize_L2(corpus_embs_np)
    torch.save({"ids": corpus_ids, "embs": torch.from_numpy(corpus_embs_np)}, cache_file)

    print(f"[INFO] Saved normalized corpus embeddings to {cache_file}")
    return corpus_ids, torch.from_numpy(corpus_embs_np).to(device)


def summarize_relation_mapping(relation_mapping, falcon_relations, falcon_qids):
    if not relation_mapping:
        return "(none)"
    summaries = []
    for rid, ents in relation_mapping.items():
        rel_label = falcon_relations.get(rid, rid)
        parts = []
        for e in ents:
            # Handle both dict-based and legacy string-based structures
            if isinstance(e, dict):
                src = falcon_qids.get(e.get("source"), e.get("source"))
                tgt = falcon_qids.get(e.get("target"), e.get("value", e.get("target")))
                parts.append(f"{src} → {tgt}")
            elif isinstance(e, str):
                parts.append(falcon_qids.get(e, e))
            else:
                parts.append(str(e))
        summaries.append(f"{rel_label} → {', '.join(parts)}")
    return "; ".join(summaries)


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

    # UPDATED TO INCLUDE ENTITY LINKING

    query_ids = list(queries.keys())
    ENRICHED_QUERIES_CACHE = "enriched_queries_webq.pkl"
    ENRICHED_RECORDS_CACHE = "enriched_records_webq.pkl"

    force_rebuild_enrichment = False

    if force_rebuild_enrichment or not (
            os.path.exists(ENRICHED_QUERIES_CACHE) and os.path.exists(ENRICHED_RECORDS_CACHE)):

        query_texts = []
        enriched_records = []
        print("[INFO] Starting entity linking enrichment and extraction...")
        for idx, qid in enumerate(query_ids):
            query_text_original = queries[qid].strip()
            query_lang = "unknown"

            try:
                query_lang = detect(query_text_original) if query_text_original else "unknown"
            except Exception:
                pass  # language detection can fail for short or non-text queries

            # ✅ Always enrich query regardless of language
            enriched_result = enrich_query_with_entities_and_facts(query_text_original)

            # print(f"Query: {query_text_original}")
            # print("Detected entities:", enriched_result.get("falcon_qids"))
            # print("Wikidata facts (raw):", enriched_result.get("wikidata_facts"))
            # print("Wikidata facts filtered:", enriched_result.get("wikidata_facts_filtered"))
            # print("DBpedia facts (raw):", enriched_result.get("dbpedia_facts_raw"))
            # print("DBpedia facts filtered:", enriched_result.get("dbpedia_facts_filtered"))

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

            # Clean up whitespace, periods, question marks, and underscores
            # query_text = re.sub(r"^[\.\s?_]+", "", query_text)  # Remove from start
            # query_text = re.sub(r"[\s?_]+", " ", query_text)  # Replace sequences with a single space

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
            if (idx + 1) % 200 == 0:  # every 20 queries
                with open(ENRICHED_QUERIES_CACHE, "wb") as f:
                    pickle.dump(query_texts, f)
                with open(ENRICHED_RECORDS_CACHE, "wb") as f:
                    pickle.dump(enriched_records, f)
                print(f"[INFO] Saved progress: {idx + 1}/{len(query_ids)} queries processed...")

            time.sleep(3) # if you want the pause

            # ✅ Final save once everything is done
            with open(ENRICHED_QUERIES_CACHE, "wb") as f:
                pickle.dump(query_texts, f)
            with open(ENRICHED_RECORDS_CACHE, "wb") as f:
                pickle.dump(enriched_records, f)
            # print(
            #     f"[INFO] Finished and saved all enriched data to {ENRICHED_QUERIES_CACHE} and {ENRICHED_RECORDS_CACHE}")
    else:
        # Load from cache to skip enrichment step
        print("[INFO] Loading preprocessed entity linking cache...")
        with open(ENRICHED_QUERIES_CACHE, "rb") as f:
            query_texts = pickle.load(f)
        with open(ENRICHED_RECORDS_CACHE, "rb") as f:
            enriched_records = pickle.load(f)
        print(f"[INFO] Loaded {len(query_texts)} cached enriched queries.")

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
        #
        # # 🟩 ADD THESE DEBUG PRINTS
        # print("\n==== DEBUG: Enriched Record ====")
        # print("Keys:", enriched.keys())
        # print("falcon_relations field:", enriched.get("falcon_relations"))
        # print("relation_entity_mapping field:", enriched.get("relation_entity_mapping"))
        # print("==============================\n")


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

    with torch.no_grad():
        printed_examples = 0  # counter for printed samples
        for i in range(0, len(query_texts), BATCH_SIZE):
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
                recall_flag = 1 if relevant & set(ranked_doc_ids) else 0
                recall_total += recall_flag

                # 🔹 Find if this query_id already exists in query_rows (from enrichment step)
                existing_row = next((r for r in query_rows if r["query_id"] == qid), None)

                if existing_row:
                    # Update the existing enriched record with MRR and Recall
                    existing_row.update({
                        "relevant_doc_ids": list(relevant),
                        f"top_{RECALL_K}_doc_ids": ranked_doc_ids[:RECALL_K],
                        "MRR": reciprocal_rank,
                        f"Recall@{RECALL_K}": recall_flag
                    })
                else:
                    # Fallback in case query_id wasn't enriched (shouldn't normally happen)
                    query_rows.append({
                        "query_id": qid,
                        "original_query": queries[qid],
                        "relevant_doc_ids": list(relevant),
                        f"top_{RECALL_K}_doc_ids": ranked_doc_ids[:RECALL_K],
                        "MRR": reciprocal_rank,
                        f"Recall@{RECALL_K}": recall_flag
                    })

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

    # ============================================================
    # 🧠 NEW SECTION: Evaluate both Original and Entity-Linked Queries
    # ============================================================

    print("\n[INFO] Running MRR comparison: Original vs Entity-Linked reformulations")

    from scipy import stats
    import numpy as np
    import matplotlib.pyplot as plt

    def compute_mrr_for_queries(query_texts_dict):
        """Helper: runs retrieval and returns per-query MRRs as dict"""
        mrr_dict = {}
        with torch.no_grad():
            for i in range(0, len(query_texts_dict), BATCH_SIZE):
                batch_items = list(query_texts_dict.items())[i:i + BATCH_SIZE]
                batch_ids = [x[0] for x in batch_items]
                batch_texts = [x[1] for x in batch_items]
                batch_embs = model.encode(batch_texts, convert_to_tensor=True).cpu().numpy().astype('float32')
                faiss.normalize_L2(batch_embs)
                D, I = index_flat.search(batch_embs, RECALL_K)
                for j, qid in enumerate(batch_ids):
                    ranked_doc_ids = [corpus_ids[idx] for idx in I[j]]
                    relevant = qrels.get(qid, set())
                    rr = 0
                    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
                        if doc_id in relevant:
                            rr = 1.0 / rank
                            break
                    mrr_dict[qid] = rr
        return mrr_dict

    # 1️⃣ Original query texts
    original_queries = {rec["query_id"]: rec["original_query"] for rec in query_rows}
    # 2️⃣ Reformulated queries (entity-linked)
    reformulated_queries = {rec["query_id"]: rec["reformulated_query"] for rec in query_rows}

    print("[INFO] Evaluating ORIGINAL queries...")
    mrr_orig_dict = compute_mrr_for_queries(original_queries)
    print("[INFO] Evaluating ENTITY-LINKED (reformulated) queries...")
    mrr_el_dict = compute_mrr_for_queries(reformulated_queries)

    # --- Merge into DataFrame ---
    df_compare = pd.DataFrame({
        "query_id": list(mrr_orig_dict.keys()),
        "MRR_original": [mrr_orig_dict[qid] for qid in mrr_orig_dict],
        "MRR_EL": [mrr_el_dict[qid] for qid in mrr_orig_dict]
    })
    df_compare["diff"] = df_compare["MRR_EL"] - df_compare["MRR_original"]

    # --- Aggregate metrics ---
    mrr_orig = df_compare["MRR_original"].mean()
    mrr_el = df_compare["MRR_EL"].mean()
    mean_diff = df_compare["diff"].mean()

    # --- Statistical tests ---
    ttest_res = stats.ttest_rel(df_compare["MRR_EL"], df_compare["MRR_original"], alternative="greater")
    wilcoxon_res = stats.wilcoxon(df_compare["MRR_EL"], df_compare["MRR_original"], alternative="greater")

    # Bootstrap 95% CI
    rng = np.random.default_rng(42)
    diffs = df_compare["diff"].values
    boot_means = [np.mean(rng.choice(diffs, len(diffs), replace=True)) for _ in range(5000)]
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    # Effect size
    cohens_d = diffs.mean() / diffs.std(ddof=1)

    print("\n==== Entity Linking Impact Analysis ====")
    print(f"MRR Original : {mrr_orig:.4f}")
    print(f"MRR EL       : {mrr_el:.4f}")
    print(f"Mean Δ (EL - Orig): {mean_diff:.4f}")
    print(f"Paired t-test (one-sided): t={ttest_res.statistic:.3f}, p={ttest_res.pvalue:.4g}")
    print(f"Wilcoxon test (one-sided): W={wilcoxon_res.statistic:.3f}, p={wilcoxon_res.pvalue:.4g}")
    print(f"Bootstrap 95% CI of Δ    : [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Cohen's d (effect size)  : {cohens_d:.3f}")

    # Save results
    df_compare.to_csv("entity_linking_per_query_mrr.csv", index=False)
    pd.DataFrame([{
        "MRR_original": mrr_orig,
        "MRR_EL": mrr_el,
        "mean_diff": mean_diff,
        "t_stat": float(ttest_res.statistic),
        "p_ttest_one_sided": float(ttest_res.pvalue),
        "wilcoxon_stat": float(wilcoxon_res.statistic),
        "p_wilcoxon_one_sided": float(wilcoxon_res.pvalue),
        "bootstrap_CI_low": ci_low,
        "bootstrap_CI_high": ci_high,
        "cohens_d": cohens_d,
        "n_queries": len(df_compare)
    }]).to_csv("entity_linking_statistical_results.csv", index=False)

    # Plot difference distribution
    plt.figure(figsize=(6, 4))
    plt.hist(df_compare["diff"], bins=30)
    plt.title("Per-query MRR Differences (EL - Original)")
    plt.xlabel("Δ MRR")
    plt.ylabel("Query count")
    plt.tight_layout()
    plt.savefig("mrr_diff_hist.png")
    plt.show()


if __name__ == "__main__":
    evaluate_webq()
