import requests
from sentence_transformers import SentenceTransformer, util
import time
import re
import json
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from entity_linking.view_results import display_full_pipeline_result
from entity_linking.intent_awareness import is_location_like_value, analyze_question_intent
from entity_linking.decompose_phrase import resolve_composite_entity, wikidata_search_label
from entity_linking.entity_linking_utils import extract_main_entity_from_question, falcon_entity_linking, dbpedia_entity_linking, merge_and_clean_entities, correct_entity_labels, wikidata_entities, fetch_entity_types, process_facts, query_wikidata_facts, query_2hop_facts, query_dbpedia_facts, query_dbpedia_2hop_facts, process_dbpedia_facts_semantic,run_sparql_query, get_entity_facts
from entity_linking.covid_handler import convert_covid_knowledge_to_sentences
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
            "premiere", "released", "open", "opening", "first performance",
            "invented", "created", "discovered", "died", "born", "birth"
        ]
    )
    is_person_question = any(
        kw in qlower for kw in [
            "who", "actor", "actress", "played", "performer",
            "portrayed", "voiced", "cast", "starring", "marry", "married"
        ]
    ) and "who is" not in qlower  # Avoid "who is" questions that might expect descriptions
    is_what_question = "what" in qlower or "which" in qlower

    # --- Helper: clean text values ---
    def clean_text(val):
        val = str(val).strip()
        val = re.sub(r'["\']', '', val)
        val = re.sub(r"\s*\(.*?\)$", "", val)
        return val.strip()

    # --- IMPROVED Date/time extraction logic ---
    def extract_date(facts):
        """Extract date values from facts, with better validation"""
        preferred_time_keywords = [
            "date", "year", "premiere", "opening", "open", "inception",
            "founded", "established", "first performance", "invented",
            "created", "discovered", "died", "death", "born", "birth",
            "start", "end", "beginning", "completion"
        ]

        iterator = facts.items() if isinstance(facts, dict) else facts

        for prop, val in iterator:
            prop_lower = str(prop).lower()
            val_str = str(val).strip()

            # Check if property indicates time/date
            if any(k in prop_lower for k in preferred_time_keywords):
                # Clean and validate the date
                cleaned = clean_date(val_str)
                if cleaned and is_valid_date(cleaned):
                    return cleaned
        return None

    def clean_date(val):
        """Extract date part from various formats"""
        val = str(val).strip()

        # Remove time portion from ISO format
        val = re.sub(r"T\d{2}:\d{2}:\d{2}Z", "", val)

        # Try to extract YYYY-MM-DD or YYYY format
        date_patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",  # YYYY-MM-DD
            r"\b\d{4}\b",  # YYYY only
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # MM/DD/YYYY
            r"\b\d{4}-\d{2}\b",  # YYYY-MM
        ]

        for pattern in date_patterns:
            match = re.search(pattern, val)
            if match:
                return match.group(0)

        return val if val else None

    def is_valid_date(date_str):
        """Validate if the string looks like a reasonable date"""
        if not date_str:
            return False

        date_str = str(date_str).strip()

        # Basic date patterns
        if re.match(r"^\d{4}$", date_str):  # YYYY
            year = int(date_str)
            return 1000 <= year <= 2100  # Reasonable year range

        if re.match(r"^\d{4}-\d{2}$", date_str):  # YYYY-MM
            year, month = map(int, date_str.split('-'))
            return 1000 <= year <= 2100 and 1 <= month <= 12

        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):  # YYYY-MM-DD
            year, month, day = map(int, date_str.split('-'))
            return (1000 <= year <= 2100 and
                    1 <= month <= 12 and
                    1 <= day <= 31)

        # MM/DD/YYYY format
        if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", date_str):
            parts = date_str.split('/')
            if len(parts) == 3:
                month, day, year = map(int, parts)
                return (1000 <= year <= 2100 and
                        1 <= month <= 12 and
                        1 <= day <= 31)

        return False

    # --- Location extraction logic ---
    def extract_location(facts):
        """Extract location-like values from facts, avoiding dates/times"""
        preferred_location_keywords = [
            "location", "place", "city", "country", "birthplace", "deathplace",
            "hometown", "residence", "headquarters", "based in", "located in"
        ]

        iterator = facts.items() if isinstance(facts, dict) else facts
        for prop, val in iterator:
            prop_lower = str(prop).lower()
            val_str = str(val).strip()

            # Skip obvious non-locations
            if (re.search(r"T\d{2}:\d{2}:\d{2}Z", val_str) or  # Timestamps
                    re.match(r"^\d{4}(-\d{2}-\d{2})?$", val_str) or  # Dates
                    re.match(r"^Point\([^)]+\)$", val_str) or  # Coordinate points
                    re.match(r"^-?\d+\.\d+$", val_str) or  # Plain numbers
                    re.match(r"^\d{4}$", val_str) or  # Years only
                    re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", val_str) or  # MM/DD/YYYY
                    len(val_str) < 2):  # Too short
                continue

            # Check if property indicates location
            if any(k in prop_lower for k in preferred_location_keywords):
                # Additional validation: value should look like a location
                if is_location_like_value(val_str):
                    return clean_text(val_str)

        return None

    # --- Person extraction logic ---
    def extract_person(facts):
        """Extract person names by looking for relationship and performance properties"""
        iterator = facts.items() if isinstance(facts, dict) else facts

        # Look for performance/relationship properties
        for prop, val in iterator:
            prop_str = str(prop).lower()
            val_str = str(val).strip()

            # Direct match for performance and relationship properties
            if any(rel in prop_str for rel in [
                "spouse", "partner", "married",
                "played by", "portrayed by", "actor", "actress", "performer"
            ]):
                return clean_text(val_str)

        return None

    # --- What-question extraction logic ---
    def extract_type_or_role(facts):
        preferred_what_keywords = [
            "occupation", "profession", "role", "type", "instance of",
            "category", "genre", "field", "discipline", "symptoms"
        ]
        iterator = facts.items() if isinstance(facts, dict) else facts
        for prop, val in iterator:
            if any(k in prop.lower() for k in preferred_what_keywords):
                return clean_text(val)
        return None

    # --- Apply extraction logic with intent awareness ---
    qlower = question_text.lower()
    is_location_question = any(kw in qlower for kw in ["where", "location", "place", "city", "country", "born", "died"])

    if is_time_question:
        # First try Wikidata
        for _, facts in wikidata_filtered_facts.items():
            date = extract_date(facts)
            if date and is_valid_date(date):
                return date
        # Then try DBpedia
        for _, facts in dbpedia_filtered_facts.items():
            date = extract_date(facts)
            if date and is_valid_date(date):
                return date
        # If no valid date found, return None to avoid wrong answers
        return None

    if is_location_question:
        # Use your intent_awareness function to validate locations
        for _, facts in wikidata_filtered_facts.items():
            location = extract_location(facts)
            if location and is_location_like_value(location):
                return location
        for _, facts in dbpedia_filtered_facts.items():
            location = extract_location(facts)
            if location and is_location_like_value(location):
                return location

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

    # CRITICAL FIX: Filter out wrong entity types BEFORE using them
    def filter_wrong_entity_types(qid_dict, question_text):
        """Filter out entities that are clearly wrong types for the question"""
        if not qid_dict:
            return {}

        filtered = {}
        wrong_types_indicators = ['news article', 'website', 'web page', 'article', 'media']

        for qid, label in qid_dict.items():
            # Skip entities that are clearly wrong types for location/person questions
            entity_types = fetch_entity_types(qid)
            type_str = ' '.join(entity_types).lower()

            # Check if this entity type makes sense for the question
            question_lower = question_text.lower()
            is_location_question = any(kw in question_lower for kw in ["where", "location", "place"])
            is_person_question = any(kw in question_lower for kw in ["who", "person", "actor"])

            should_keep = True

            # For location/person questions, reject news articles and websites
            if (is_location_question or is_person_question) and any(
                    wrong_type in type_str for wrong_type in wrong_types_indicators):
                print(f"[ENTITY FILTER] Removing {qid} ('{label}') - wrong type: {entity_types}")
                should_keep = False

            # Also reject entities with labels that are clearly question phrases
            if len(label.split()) > 3 and any(
                    word in label.lower() for word in ['go to', 'where did', 'when did', 'who did']):
                print(f"[ENTITY FILTER] Removing {qid} ('{label}') - appears to be question phrase")
                should_keep = False

            if should_keep:
                filtered[qid] = label

        return filtered

    # Apply filtering
    if qids:
        filtered_qids = filter_wrong_entity_types(qids, original_text)
        falcon_qids.update(filtered_qids)

        # If filtering removed all entities, try to find the real entity
        if not falcon_qids and original_text:
            print("[ENTITY RECOVERY] No valid entities found, attempting to extract main entity...")
            # Try to extract the main person/location from the question
            main_entity = extract_main_entity_from_question(original_text)
            if main_entity:
                print(f"[ENTITY RECOVERY] Searching for: {main_entity}")
                # Use Wikidata search to find the correct entity
                search_results = wikidata_search_label(main_entity, search_type="item", limit=3)
                if search_results:
                    best_result = search_results[0]
                    falcon_qids[best_result['id']] = best_result['label']
                    print(f"[ENTITY RECOVERY] Found: {best_result['id']} - {best_result['label']}")

    # ========== COVID-19 DETECTION & KNOWLEDGE EXTRACTION ==========
    def is_covid_related_question(question_text, entities_dict):
        """Check if question is COVID-19 related"""
        covid_keywords = [
            "covid", "coronavirus", "sars-cov-2", "pandemic", "vaccine",
            "variant", "mutation", "symptom", "lockdown", "quarantine",
            "social distancing", "mask", "ventilator", "outbreak", "epidemic"
        ]

        question_lower = question_text.lower()

        # Check question text
        if any(kw in question_lower for kw in covid_keywords):
            return True

        # Check entity labels
        for label in entities_dict.values():
            label_lower = label.lower()
            if any(kw in label_lower for kw in covid_keywords):
                return True

        return False

    # Check if this is a COVID-19 related question
    is_covid_question = is_covid_related_question(original_text, falcon_qids)
    covid_knowledge = {}

    if is_covid_question:
        print("[COVID DETECTION] COVID-19 related question detected, extracting specialized knowledge...")
        try:
            # Import and use the COVID handler
            from entity_linking.covid_handler import extract_covid_knowledge_from_wikidata
            covid_knowledge = extract_covid_knowledge_from_wikidata(original_text, falcon_qids)
            print(f"[COVID KNOWLEDGE] Extracted {sum(len(items) for items in covid_knowledge.values())} COVID facts")
        except Exception as e:
            print(f"[COVID HANDLER ERROR] Failed to extract COVID knowledge: {e}")
            covid_knowledge = {}

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
        print(f"[RELATION DEBUG] Processing {len(falcon_relations)} relations: {falcon_relations}")
        for rid, rlabel in falcon_relations.items():
            for qid, qlabel in falcon_qids.items():
                # print(f"[RELATION DEBUG] Processing relation {rid} ('{rlabel}') for entity {qid} ('{qlabel}')")

                # 1️⃣ Try to get forward facts first
                # print(f"[RELATION DEBUG] Looking for forward facts: {rid} for {qid}")
                rel_specific = get_entity_facts(qid, property_filter=rid)
                # print(f"[RELATION DEBUG] Forward facts found: {len(rel_specific)}")
                for fact in rel_specific:
                    print(f"[RELATION DEBUG] Forward fact: {fact}")

                # 2️⃣ If no forward facts found, try the correct character-actor properties
                if not rel_specific:
                    # print(f"[RELATION DEBUG] Trying character-actor relationships for {qid}")

                    # Try multiple properties that link characters to actors
                    character_actor_queries = [
                        # P161 - cast member (most common for characters)
                        f"""
                        SELECT ?actor ?actorLabel WHERE {{
                            wd:{qid} wdt:P161 ?actor.
                            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                        }}
                        LIMIT 10
                        """,
                        # P725 - voice actor (for animated characters)
                        f"""
                        SELECT ?actor ?actorLabel WHERE {{
                            wd:{qid} wdt:P725 ?actor.
                            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                        }}
                        LIMIT 10
                        """,
                        # P175 - performer (original relation, but try as forward)
                        f"""
                        SELECT ?actor ?actorLabel WHERE {{
                            wd:{qid} wdt:P175 ?actor.
                            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                        }}
                        LIMIT 10
                        """
                    ]

                    for i, query in enumerate(character_actor_queries):
                        try:
                            # print(f"[RELATION DEBUG] Trying character-actor query {i + 1}")
                            results = run_sparql_query(query)
                            # print(f"[RELATION DEBUG] Character-actor query {i + 1} returned {len(results)} results")

                            for res in results:
                                actor_label = res.get("actorLabel", "")
                                actor_qid = res.get("actor", "").split("/")[-1] if res.get("actor") else ""

                                if actor_label and len(actor_label.strip()) > 1:
                                    rel_specific.append({
                                        "property": f"played by (character-actor)",
                                        "value": actor_label,
                                        "target_qid": actor_qid,
                                        "provenance": f"character-actor-{i + 1}"
                                    })
                                    # print(f"[RELATION DEBUG] Found actor: {actor_label}")

                            # If we found results, break out of the loop
                            if rel_specific:
                                break

                        except Exception as e:
                            print(f"[WARN] Character-actor query {i + 1} failed: {e}")

                # 3️⃣ Record found relations
                if rel_specific:
                    # print(f"[RELATION SUCCESS] Found {len(rel_specific)} actors for character {qlabel}")
                    for f in rel_specific:
                        f["provenance"] = f.get("provenance", "relation-filter")
                        relation_entity_mapping.setdefault(rid, []).append({
                            "source": qid,
                            "target": f.get("target_qid"),
                            "value": f.get("value", "")
                        })

                    relation_facts.setdefault(rid, []).extend(rel_specific)

                    # Add relation facts to ALL fact collections for answer extraction
                    for fact in rel_specific:
                        # Add to combined facts
                        qid_facts_combined.setdefault(qid, []).append({
                            "property": fact["property"],
                            "value": fact["value"],
                            "provenance": fact["provenance"]
                        })

                        # CRITICAL: Also add to filtered facts as tuple format for direct answer extraction
                        qid_facts_filtered.setdefault(qid, []).append(
                            (fact["property"], fact["value"])
                        )

                        # Also add to readable sentences for reformulation
                        readable_sentences.setdefault(qid, []).append(
                            f"{qlabel} was played by {fact['value']}."
                        )

                    print(f"[RELATION DEBUG] Added {len(rel_specific)} relation facts to filtered facts for {qid}")

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

    # ========== ADD COVID KNOWLEDGE TO REFORMULATION ==========
    if is_covid_question and covid_knowledge:
        print("[COVID REFORMULATION] Adding COVID knowledge to query reformulation...")
        covid_sentences = convert_covid_knowledge_to_sentences(covid_knowledge)

        # Debug: Show what COVID sentences were generated
        print(f"[COVID REFORMULATION] Generated {len(covid_sentences)} COVID sentences")
        for i, sentence in enumerate(covid_sentences):
            print(f"[COVID REFORMULATION] Sentence {i + 1}: {sentence}")

        # SPECIAL HANDLING FOR COVID QUESTIONS: Always prioritize COVID knowledge
        if covid_sentences:
            # For COVID questions, use a lower similarity threshold or direct inclusion
            ranked_covid_sentences = rank_sentences_by_similarity(original_text, covid_sentences, min_score=0.4)
            print(
                f"[COVID REFORMULATION] After ranking: {len(ranked_covid_sentences)} COVID sentences above threshold (0.4)")

            if ranked_covid_sentences:
                # Add ALL ranked COVID sentences to the beginning of expanded_sentences for priority
                expanded_sentences = ranked_covid_sentences + expanded_sentences
                print(
                    f"[COVID REFORMULATION] Added {len(ranked_covid_sentences)} COVID sentences to reformulation (prioritized)")
            else:
                # If no sentences pass even lower threshold, add top 2 COVID sentences anyway
                print("[COVID REFORMULATION] No COVID sentences passed threshold, adding top 2 directly")
                expanded_sentences = covid_sentences[:2] + expanded_sentences

    q_emb = embedding_model.encode(original_text, convert_to_tensor=True)
    # Use lower threshold for COVID questions to ensure COVID knowledge is included
    similarity_threshold = 0.4 if is_covid_question else 0.5
    ranked_expanded = [(util.cos_sim(q_emb, embedding_model.encode(s, convert_to_tensor=True)).item(), s)
                       for s in expanded_sentences]
    ranked_expanded = [s for score, s in sorted(ranked_expanded, reverse=True) if score >= similarity_threshold]
    top_sentences = ranked_expanded[:1]



    # ========== UNIFIED REFORMULATION STRATEGY ==========
    def build_covid_reformulation(question, covid_knowledge, expanded_sentences):
        """Build COVID-specific reformulation with prioritized knowledge"""
        print("[COVID REFORMULATION] Building COVID-focused reformulation...")

        qlower = question.lower()

        # 1. Mutation-specific reformulation
        if "mutation" in qlower:
            mutation_names = []
            for category in ["mutations", "spike_mutations"]:
                if category in covid_knowledge:
                    for mutation in covid_knowledge[category]:
                        name = mutation.get('name', '')
                        if name:
                            # Clean mutation name
                            clean_name = re.sub(r'\s*\(.*?\)', '', name).strip()
                            clean_name = re.sub(r'\s*mutation\s*', '', clean_name, flags=re.IGNORECASE).strip()
                            if clean_name and clean_name not in mutation_names:
                                mutation_names.append(clean_name)

            if mutation_names:
                unique_mutations = list(set(mutation_names))[:3]
                mutation_text = ", ".join(unique_mutations)
                return f"{question} {mutation_text}."

        # 2. Symptom-specific reformulation
        elif "symptom" in qlower:
            symptom_names = []
            if "symptoms" in covid_knowledge:
                for symptom in covid_knowledge["symptoms"]:
                    name = symptom.get('name', '')
                    if name and name.lower() not in ["unknown", ""]:
                        symptom_names.append(name)

            if symptom_names:
                top_symptoms = symptom_names[:min(5, len(symptom_names))]
                symptoms_text = ", ".join(top_symptoms)
                return f"{question} Common symptoms: {symptoms_text}."

        # 3. Vaccine-specific reformulation
        elif any(kw in qlower for kw in ["vaccine", "immunization", "pfizer", "moderna"]):
            vaccine_info = []
            if "vaccines" in covid_knowledge:
                for vaccine in covid_knowledge["vaccines"]:
                    name = vaccine.get('name', '')
                    if name:
                        vaccine_info.append(name)

            if vaccine_info:
                vaccines_text = ", ".join(vaccine_info[:3])
                return f"{question} Vaccines include: {vaccines_text}."

        # 4. General COVID reformulation - use top COVID sentences
        if expanded_sentences:
            # Filter for COVID-related sentences (they should be prioritized already)
            covid_sentences = [s for s in expanded_sentences if any(
                covid_kw in s.lower() for covid_kw in ["covid", "coronavirus", "sars", "pandemic"]
            )]

            if covid_sentences:
                context = covid_sentences[0] if covid_sentences else expanded_sentences[0]
                return f"{question} {context}"

        # 5. Fallback to generic COVID context
        return f"{question} COVID-19 related information"

    def build_general_reformulation(question, direct_answer, top_sentences, qid_facts_filtered, dbpedia_facts_filtered):
        """Build reformulation for non-COVID questions"""
        qlower = question.lower()
        is_time_question = any(kw in qlower for kw in ["when", "date", "year", "founded", "established"])

        # Try direct answer first
        if direct_answer:
            ans = direct_answer.strip()
            ans = re.sub(r"T\d{2}:\d{2}:\d{2}Z", "", ans)  # Clean timestamps

            if re.match(r"^\d{4}(-\d{2}-\d{2})?$", ans):
                return f"{question} {ans}"
            else:
                return f"{question} {ans}"

        # For time questions with no answer, don't append wrong info
        elif is_time_question:
            return question.strip()

        # Use top semantic sentences
        elif top_sentences:
            best_sentence = top_sentences[0].strip()
            answer_match = re.search(r"is\s+(.+?)\.*$", best_sentence)
            if answer_match:
                short_answer = re.sub(r"\.$", "", answer_match.group(1).strip())
            else:
                short_answer = re.sub(r"\.$", "", best_sentence.split()[-1].strip())
            return f"{question} {short_answer}".strip()

        # Fallback
        else:
            return question.strip()

    # ========== APPLY REFORMULATION STRATEGY ==========
    if is_covid_question and covid_knowledge:
        reformulated_query = build_covid_reformulation(original_text, covid_knowledge, expanded_sentences)
        print(f"[COVID REFORMULATION] Final: {reformulated_query}")
    else:
        direct_answer = find_direct_answer_from_facts(original_text, qid_facts_filtered, dbpedia_facts_filtered)
        reformulated_query = build_general_reformulation(
            original_text, direct_answer, top_sentences, qid_facts_filtered, dbpedia_facts_filtered
        )
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
        "is_covid_related": is_covid_question,
        "covid_knowledge": covid_knowledge,
    }

    return result


# # -----------------------------
# # Example Usage
# # # # -----------------------------
# prompt = "what are the initial symptoms of covid 19 viral pneumonia"
# result = enrich_query_with_entities_and_facts(prompt)
# # print_clean_pipeline_result(result)
# #
# display_full_pipeline_result(result,1)