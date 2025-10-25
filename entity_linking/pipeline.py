import requests
from sentence_transformers import SentenceTransformer, util
import time
import re
import json
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from entity_linking.view_results import display_full_pipeline_result
from entity_linking.decompose_phrase import resolve_composite_entity
from entity_linking.entity_linking_utils import falcon_entity_linking, dbpedia_entity_linking, merge_and_clean_entities, correct_entity_labels, wikidata_entities, fetch_entity_types, process_facts, query_wikidata_facts, query_2hop_facts, query_dbpedia_facts, query_dbpedia_2hop_facts, process_dbpedia_facts_semantic,run_sparql_query, get_entity_facts
# Load SentenceTransformer model for semantic similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"

# Prototype embedding for generic properties
generic_proto_emb = embedding_model.encode("general topic or category", convert_to_tensor=True)
GENERIC_THRESHOLD = 0.65  # similarity threshold to consider property as generic

# --- New helper: try to extract direct location/answer from filtered facts ---
def find_direct_answer_from_facts(question_text, wikidata_filtered_facts, dbpedia_filtered_facts):
    """
    Extracts direct answers (date/time for 'when', people for 'who', occupation/type for 'what')
    from filtered Wikidata and DBpedia facts.
    Works for both dict and list-of-tuples fact formats.
    """
    import re

    qlower = question_text.lower()

    # Detect question types
    is_time_question = any(
        kw in qlower for kw in [
            "when", "date", "year", "founded", "established",
            "premiere", "released", "open", "opening", "first performance"
        ]
    )
    is_person_question = any(
        kw in qlower for kw in [
            "who", "actor", "actress", "played", "performer",
            "portrayed", "voiced", "cast", "starring"
        ]
    )
    is_what_question = "what" in qlower or "which" in qlower

    # --- Helper: clean text values ---
    def clean_text(val):
        val = str(val).strip()
        val = re.sub(r'["\']', '', val)
        val = re.sub(r"\s*\(.*?\)$", "", val)
        return val.strip()

    # --- Date/time extraction logic ---
    def clean_date(val):
        val = str(val).strip()
        val = re.sub(r"T\d{2}:\d{2}:\d{2}Z", "", val)
        m = re.search(r"\d{4}(-\d{2}-\d{2})?", val)
        return m.group(0) if m else val

    def extract_date(facts):
        preferred_time_keywords = [
            "date", "year", "premiere", "opening", "open", "inception",
            "founded", "established", "first performance"
        ]
        iterator = facts.items() if isinstance(facts, dict) else facts
        for prop, val in iterator:
            if any(k in prop.lower() for k in preferred_time_keywords):
                cleaned = clean_date(val)
                if cleaned:
                    return cleaned
        return None

    # --- Person extraction logic ---
    def extract_person(facts):
        preferred_person_keywords = [
            "performer", "actor", "actress", "cast", "starring",
            "played by", "portrayed by", "voice actor", "narrator"
        ]
        iterator = facts.items() if isinstance(facts, dict) else facts
        for prop, val in iterator:
            if any(k in prop.lower() for k in preferred_person_keywords):
                return clean_text(val)
        return None

    # --- What-question extraction logic ---
    def extract_type_or_role(facts):
        preferred_what_keywords = [
            "occupation", "profession", "role", "type", "instance of",
            "category", "genre", "field", "discipline"
        ]
        iterator = facts.items() if isinstance(facts, dict) else facts
        for prop, val in iterator:
            if any(k in prop.lower() for k in preferred_what_keywords):
                return clean_text(val)
        return None

    # --- Apply extraction logic ---
    if is_time_question:
        for _, facts in wikidata_filtered_facts.items():
            date = extract_date(facts)
            if date:
                return date
        for _, facts in dbpedia_filtered_facts.items():
            date = extract_date(facts)
            if date:
                return date

    if is_person_question:
        for _, facts in wikidata_filtered_facts.items():
            person = extract_person(facts)
            if person:
                return person
        for _, facts in dbpedia_filtered_facts.items():
            person = extract_person(facts)
            if person:
                return person

    if is_what_question:
        for _, facts in wikidata_filtered_facts.items():
            role = extract_type_or_role(facts)
            if role:
                return role
        for _, facts in dbpedia_filtered_facts.items():
            role = extract_type_or_role(facts)
            if role:
                return role

    return None


def enrich_query_with_entities_and_facts(original_text):
    # ------------------- Entity Linking -------------------
    falcon_qids = {}
    falcon_dbpedia_entities = {}

    qids, dbs, qrels, db_rels = falcon_entity_linking(original_text)

    if qids:
        falcon_qids.update(qids)
    if dbs:
        falcon_dbpedia_entities.update(dbs)

    # --- Robust relation extraction ---
    falcon_relations = {}

    def clean_relations(rel_dict):
        clean_map = {}
        if isinstance(rel_dict, dict):
            for k, v in rel_dict.items():
                if not k or not v:
                    continue
                # Only accept keys that look like relation IDs or URLs
                if isinstance(k, str) and (k.startswith("P") or "ontology" in k or "property" in k):
                    if isinstance(v, str) and len(v.strip()) > 2:
                        clean_map[k] = v.strip()
        elif isinstance(rel_dict, list):
            for rel in rel_dict:
                if isinstance(rel, dict):
                    key = rel.get("relation") or rel.get("uri") or rel.get("id")
                    label = rel.get("label") or rel.get("surface_form") or rel.get("text")
                    if key and label and len(label.strip()) > 2:
                        clean_map[key] = label.strip()
        return clean_map

    falcon_relations.update(clean_relations(qrels))
    falcon_relations.update(clean_relations(db_rels))

    # print("\n=== DEBUG: FALCON RELATIONS CLEANED ===")
    # print(json.dumps(falcon_relations, indent=2))

    # ------------------- DBpedia Entity Linking -------------------
    dbpedia_spotlight = dbpedia_entity_linking(original_text)
    dbpedia_entities = merge_and_clean_entities(falcon_dbpedia_entities, dbpedia_spotlight)

    # Semantic deduplication (same as before)
    all_labels = list(dbpedia_entities.values()) + list(falcon_dbpedia_entities.values())#
    unique_labels = {}
    for label in all_labels:
        emb = embedding_model.encode(label, convert_to_tensor=True)
        if not any(util.cos_sim(emb, existing_emb).item() > 0.8 for existing_emb in unique_labels.values()):
            unique_labels[label] = emb
    dbpedia_entities = {uri: label for uri, label in dbpedia_entities.items() if label in unique_labels}

    # Merge overlapping entity labels
    merged_dbpedia = {}
    for uri, label in dbpedia_entities.items():
        if not any(label.lower() in existing_label.lower() or existing_label.lower() in label.lower()
                   for existing_label in merged_dbpedia.values()):
            merged_dbpedia[uri] = label
    dbpedia_entities = merged_dbpedia

    falcon_qids = correct_entity_labels(falcon_qids)
    # Try to detect composite phrase QIDs like "state flower of arizona"
    falcon_qids, composite_mappings = resolve_composite_entity(falcon_qids)
    if composite_mappings:
        print("[INFO] Composite mappings found:", json.dumps(composite_mappings, indent=2))
    dbpedia_entities = correct_entity_labels(dbpedia_entities)

    # ------------------- Wikidata Entities -------------------
    qid_to_entities = {}
    for qid, label in falcon_qids.items():
        entity_forms = wikidata_entities(qid) or [label]
        qid_to_entities[qid] = entity_forms

    # ------------------- Entity Types -------------------
    entity_types = {qid: fetch_entity_types(qid) for qid in falcon_qids}

    # ------------------- Wikidata & DBpedia Facts -------------------
    (qid_facts_raw, qid_facts_2hop, qid_facts_combined,
     qid_facts_filtered, readable_sentences) = process_facts(
        falcon_qids,
        query_wikidata_facts,
        query_2hop_facts,
        label_func=lambda qid: falcon_qids[qid],
        question=original_text,
        entity_types=entity_types
    )


    # --- NEW: Enrich using composite mappings (via DBpedia + Wikidata fallback) ---
    if composite_mappings:
        print("[INFO] Running composite QID enrichment (DBpedia + Wikidata fallback)...")

        for orig_cqid, comp in composite_mappings.items():
            prop_label = comp.get("prop_label")
            entity_label = comp.get("entity_label")
            resolved_qid = comp.get("resolved_qid")
            prop_values = comp.get("property_values", [])

            # ✅ 1️⃣ If resolve_composite_entity() already found DBpedia values, use them directly
            if prop_values:
                for val_uri, val_label in prop_values:
                    synthetic_sentence = f"{entity_label}'s {prop_label} is {val_label}."
                    print(f"[COMPOSITE RESULT] {synthetic_sentence}")

                    qid_facts_combined.setdefault(resolved_qid, []).append({
                        "property": prop_label,
                        "value": val_label,
                        "uri": val_uri,
                        "provenance": "dbpedia-direct"
                    })
                    readable_sentences.setdefault(resolved_qid, []).append(synthetic_sentence)
                continue  # skip further enrichment if we already have DBpedia result

            # ✅ 2️⃣ Otherwise, fall back to enrichment as before
            print(f"[COMPOSITE ENRICH] Trying to answer '{prop_label} of {entity_label}'")

            matched_dbpedia_uri = None
            for uri, label in dbpedia_entities.items():
                if label.lower() == entity_label.lower() or entity_label.lower() in label.lower():
                    matched_dbpedia_uri = uri
                    break

            if not matched_dbpedia_uri:
                print(f"[WARN] No DBpedia entity found for '{entity_label}', skipping.")
                continue

            all_facts = query_dbpedia_facts(matched_dbpedia_uri)
            all_facts += query_dbpedia_2hop_facts(matched_dbpedia_uri)

            if not all_facts:
                print(f"[WARN] No DBpedia facts found for {entity_label}")
                continue

            prop_emb = embedding_model.encode(prop_label, convert_to_tensor=True)
            scored = []
            for p, v in all_facts:
                if not p or not v:
                    continue
                p_emb = embedding_model.encode(p, convert_to_tensor=True)
                v_emb = embedding_model.encode(v, convert_to_tensor=True)
                score = max(util.cos_sim(prop_emb, p_emb).item(), util.cos_sim(prop_emb, v_emb).item())
                if score >= 0.45:
                    scored.append((score, p, v))

            if not scored:
                print(f"[INFO] No semantically close DBpedia facts for '{prop_label}' in {entity_label}")
                continue

            scored.sort(reverse=True, key=lambda x: x[0])
            best_fact = scored[0][1:]

            synthetic_sentence = f"{entity_label}'s {prop_label} is {best_fact[1]}."
            print(f"[COMPOSITE RESULT] {synthetic_sentence}")

            qid_facts_combined.setdefault(resolved_qid, []).append({
                "property": prop_label,
                "value": best_fact[1],
                "provenance": "dbpedia-composite"
            })
            readable_sentences.setdefault(resolved_qid, []).append(synthetic_sentence)

    # ------------------- DBpedia Facts -------------------
    dbpedia_entities_labels = {uri: [label] for uri, label in dbpedia_entities.items()}
    dbpedia_facts_raw, dbpedia_facts_2hop, dbpedia_facts_combined = {}, {}, {}

    for uri, label in dbpedia_entities.items():
        raw = query_dbpedia_facts(uri)
        hop2 = query_dbpedia_2hop_facts(uri)
        combined = list(set(raw + hop2))
        dbpedia_facts_raw[uri] = raw
        dbpedia_facts_2hop[uri] = hop2
        dbpedia_facts_combined[uri] = combined

    dbpedia_facts_filtered, dbpedia_readable_sentences = process_dbpedia_facts_semantic(
        dbpedia_entities, question=original_text
    )

    # ------------------- Relation-Aware Enrichment -------------------
    relation_facts = {}
    relation_entity_mapping = {}

    if falcon_relations:
        for rid, rlabel in falcon_relations.items():
            for qid, qlabel in falcon_qids.items():

                # 1️⃣ Try to get forward facts first
                rel_specific = get_entity_facts(qid, property_filter=rid)

                # 2️⃣ If none found, automatically try the reverse direction
                if not rel_specific:
                    try:
                        reverse_query = f"""
                        SELECT ?target ?targetLabel WHERE {{
                            ?target wdt:{rid} wd:{qid}.
                            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                        }}
                        """
                        reverse_results = run_sparql_query(reverse_query)
                        rel_specific = []
                        for res in reverse_results:
                            rel_specific.append({
                                "property": rid,
                                "value": res.get("targetLabel", ""),
                                "target_qid": res.get("target", ""),
                                "provenance": "reverse-relation"
                            })
                    except Exception as e:
                        print(f"[WARN] Reverse relation query failed for {rid}/{qid}: {e}")

                # 3️⃣ Record found relations
                if rel_specific:
                    for f in rel_specific:
                        f["provenance"] = f.get("provenance", "relation-filter")
                        relation_entity_mapping.setdefault(rid, []).append({
                            "source": qid,
                            "target": f.get("target_qid"),
                            "value": f.get("value", "")
                        })
                    relation_facts.setdefault(rid, []).extend(rel_specific)
                    qid_facts_combined.setdefault(qid, []).extend(rel_specific)

    # ------------------- Reformulated Query -------------------
    def rank_sentences_by_similarity(question, sentences, min_score=0.65):
        q_emb = embedding_model.encode(question, convert_to_tensor=True)
        scored = [(util.cos_sim(q_emb, embedding_model.encode(s, convert_to_tensor=True)).item(), s)
                  for s in sentences]
        return [s for score, s in sorted(scored, reverse=True) if score >= min_score]

    # ------------------- Reformulated Query (improved for location questions) -------------------
    def rank_sentences_by_similarity(question, sentences, min_score=0.65):
        q_emb = embedding_model.encode(question, convert_to_tensor=True)
        scored = [(util.cos_sim(q_emb, embedding_model.encode(s, convert_to_tensor=True)).item(), s)
                  for s in sentences]
        return [s for score, s in sorted(scored, reverse=True) if score >= min_score]

    expanded_sentences = []
    for sentences in readable_sentences.values():
        expanded_sentences.extend(rank_sentences_by_similarity(original_text, sentences))
    for sentences in dbpedia_readable_sentences.values():
        expanded_sentences.extend(rank_sentences_by_similarity(original_text, sentences))

    q_emb = embedding_model.encode(original_text, convert_to_tensor=True)
    ranked_expanded = [(util.cos_sim(q_emb, embedding_model.encode(s, convert_to_tensor=True)).item(), s)
                       for s in expanded_sentences]
    ranked_expanded = [s for score, s in sorted(ranked_expanded, reverse=True) if score >= 0.5]
    top_sentences = ranked_expanded[:1]




    # Try to prefer a direct place-of-birth answer from filtered facts
    direct_answer = find_direct_answer_from_facts(original_text, qid_facts_filtered, dbpedia_facts_filtered)

    if direct_answer:
        ans = direct_answer.strip()

        # Clean timestamps (e.g., 1977-04-21T00:00:00Z → 1977-04-21)
        ans = re.sub(r"T\d{2}:\d{2}:\d{2}Z", "", ans)

        # If we got a valid date or year, append it directly to the query
        if re.match(r"^\d{4}(-\d{2}-\d{2})?$", ans):
            reformulated_query = f"{original_text} {ans}"
        else:
            reformulated_query = f"{original_text} {ans}"
    else:
        if top_sentences:
            best_sentence = top_sentences[0].strip()
            answer_match = re.search(r"is\s+(.+?)\.*$", best_sentence)
            if answer_match:
                short_answer = re.sub(r"\.$", "", answer_match.group(1).strip())
            else:
                short_answer = re.sub(r"\.$", "", best_sentence.split()[-1].strip())
            reformulated_query = f"{original_text} {short_answer}".strip()
        else:
            reformulated_query = original_text.strip()

    # ------------------- Build Result -------------------
    result = {
        "original_query": original_text,
        "falcon_qids": falcon_qids,
        "wikidata_entities": qid_to_entities,
        "dbpedia_entities": dbpedia_entities,
        "dbpedia_entities_labels": dbpedia_entities_labels,
        "wikidata_facts": qid_facts_raw,
        "wikidata_facts_2hop": qid_facts_2hop,
        "wikidata_facts_combined": qid_facts_combined,
        "wikidata_facts_filtered": qid_facts_filtered,
        "dbpedia_facts_raw": dbpedia_facts_raw,
        "dbpedia_facts_2hop": dbpedia_facts_2hop,
        "dbpedia_facts_combined": dbpedia_facts_combined,
        "dbpedia_facts_filtered": dbpedia_facts_filtered,
        "reformulated_query": reformulated_query.strip(),
        "falcon_relations": falcon_relations,
        "relation_facts": relation_facts,
        "relation_entity_mapping": relation_entity_mapping,
        "natural_language_summary": {**readable_sentences, **dbpedia_readable_sentences},
        "entity_types": entity_types,
    }

    return result


# # -----------------------------
# # Example Usage
# # # # -----------------------------
# prompt = "who did cam newton sign with?"
# result = enrich_query_with_entities_and_facts(prompt)
# # print_clean_pipeline_result(result)
# #
# display_full_pipeline_result(result,1)