import requests
from sentence_transformers import SentenceTransformer, util
import time
import re
import json
from SPARQLWrapper import SPARQLWrapper, JSON
# Load SentenceTransformer model for semantic similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"

# Prototype embedding for generic properties
generic_proto_emb = embedding_model.encode("general topic or category", convert_to_tensor=True)
GENERIC_THRESHOLD = 0.65  # similarity threshold to consider property as generic


# -----------------------------
# 1. Falcon Entity Linking
# -----------------------------
import requests

def falcon_entity_linking(text):
    """
    Use Falcon 2.0 API to detect and link entities and relations from the input text.
    Returns four dictionaries:
      - wikidata_entities: {QID: surface_form}
      - dbpedia_entities: {DBpedia_URI: surface_form}
      - wikidata_relations: {PID: surface_form}
      - dbpedia_relations: {DBpedia_URI: surface_form}
    """
    url = "https://labs.tib.eu/falcon/falcon2/api?mode=long"
    payload = {"text": text}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()

        # Extract entity info
        entities_wd = result.get("entities_wikidata", [])
        entities_db = result.get("entities_dbpedia", [])

        wikidata_entities = {
            e["URI"].split("/")[-1]: e.get("surface form", "")
            for e in entities_wd if "URI" in e
        }
        dbpedia_entities = {
            e["URI"]: e.get("surface form", "")
            for e in entities_db if "URI" in e
        }

        # Extract relation info
        relations_wd = result.get("relations_wikidata", [])
        relations_db = result.get("relations_dbpedia", [])

        wikidata_relations = {
            r["URI"].split("/")[-1]: r.get("surface form", "")
            for r in relations_wd if "URI" in r
        }
        dbpedia_relations = {
            r["URI"]: r.get("surface form", "")
            for r in relations_db if "URI" in r
        }
        print(wikidata_entities, dbpedia_entities, wikidata_relations, dbpedia_relations)
        return wikidata_entities, dbpedia_entities, wikidata_relations, dbpedia_relations

    except requests.exceptions.RequestException as e:
        print(f"[Falcon] Request error: {e}")
        return {}, {}, {}, {}
    except ValueError:
        print("[Falcon] Response parsing error")
        return {}, {}, {}, {}





# -----------------------------
# 2. DBpedia Entity Linking (fallback)
# -----------------------------
def dbpedia_entity_linking(text, confidence=0.5, support=20):
    """
    Extract DBpedia entities from text using DBpedia Spotlight API.
    """
    url = "https://api.dbpedia-spotlight.org/en/annotate"
    headers = {"Accept": "application/json"}
    params = {
        "text": text,
        "confidence": confidence,
        "support": support
    }

    response = requests.get(url, headers=headers, params=params)
    entities = {}
    if response.status_code == 200:
        data = response.json()
        resources = data.get("Resources", [])
        for res in resources:
            uri = res["@URI"]
            surface_form = res["@surfaceForm"]
            entities[uri] = surface_form
    else:
        print(f"[DBpedia] Error: {response.status_code}")
    return entities


# -----------------------------
# 3. Wikidata Utilities
# -----------------------------
def wikidata_entities(qid):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = response.json()
    entity_data = data['entities'].get(qid, {})
    entities = []

    labels = entity_data.get('labels', {})
    if 'en' in labels:
        entities.append(labels['en']['value'])

    alias_data = entity_data.get('entities', {})
    if 'en' in alias_data:
        entities.extend([alias['value'] for alias in alias_data['en']])

    return list(set(entities))



def query_wikidata_facts(qid):
    query = f"""
    SELECT ?propertyLabel ?valueLabel WHERE {{
      wd:{qid} ?p ?value .
      ?property wikibase:directClaim ?p .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 50
    """
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(WIKIDATA_SPARQL_ENDPOINT, params={"query": query}, headers=headers)

    # Handle rate limit
    if response.status_code == 429:
        print(f"[WARN] 429 Too Many Requests for {qid}, retrying after 1s...")
        time.sleep(1)
        response = requests.get(WIKIDATA_SPARQL_ENDPOINT, params={"query": query}, headers=headers)

    # --- SAFE JSON HANDLING PATCH START ---
    if not response.ok:
        print(f"[WARN] Wikidata HTTP error {response.status_code} for {qid}")
        print("Response text (first 200 chars):", response.text[:200])
        return []

    try:
        data = response.json()
    except Exception as e:
        print(f"[WARN] Failed to parse JSON for {qid}: {e}")
        print(f"Status: {response.status_code}")
        print("Raw response (first 200 chars):", response.text[:200])
        return []
    # --- SAFE JSON HANDLING PATCH END ---

    results = data.get('results', {}).get('bindings', [])
    facts = []
    for result in results:
        prop = result.get('propertyLabel', {}).get('value')
        val = result.get('valueLabel', {}).get('value')
        if prop and val:
            facts.append((prop, val))
    return facts


def fetch_entity_types(qid):
    """
    Fetch the 'instance of' and 'occupation' types for a Wikidata entity
    Returns a list of type labels
    """
    query = f"""
    SELECT ?typeLabel WHERE {{
      wd:{qid} wdt:P31|wdt:P106 ?type .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 10
    """
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(WIKIDATA_SPARQL_ENDPOINT, params={"query": query}, headers=headers)
    if response.status_code != 200:
        return []
    results = response.json().get('results', {}).get('bindings', [])
    types = [r['typeLabel']['value'] for r in results if 'typeLabel' in r]
    return types


def query_2hop_facts(qid):
    query = f"""
    SELECT ?pLabel ?oLabel ?p2Label ?o2Label WHERE {{
      wd:{qid} ?p1 ?o .
      ?p wikibase:directClaim ?p1 .
      OPTIONAL {{ ?o ?p2 ?o2 .
                 ?p2 rdfs:label ?p2Label . FILTER (lang(?p2Label) = "en")
                 ?o2 rdfs:label ?o2Label . FILTER (lang(?o2Label) = "en") }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 50
    """
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(WIKIDATA_SPARQL_ENDPOINT, params={"query": query}, headers=headers)
    if response.status_code != 200:
        print(f"[SPARQL] 2-hop Query failed for {qid} with status {response.status_code}")
        return []

    results = response.json().get('results', {}).get('bindings', [])
    facts = []
    for result in results:
        prop = result.get('pLabel', {}).get('value')
        val = result.get('oLabel', {}).get('value')
        prop2 = result.get('p2Label', {}).get('value')
        val2 = result.get('o2Label', {}).get('value')
        if prop and val:
            facts.append((prop, val))
        if prop2 and val2:
            facts.append((prop2, val2))
    return facts


def query_dbpedia_facts(uri):
    """
    Query DBpedia 1-hop facts for a given entity URI.
    """
    query = f"""
    SELECT ?property ?value WHERE {{
        <{uri}> ?property ?value .
        FILTER (lang(?value) = 'en' || datatype(?value) = xsd:string)
    }}
    LIMIT 50
    """
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(DBPEDIA_SPARQL_ENDPOINT, params={"query": query}, headers=headers)
    if response.status_code != 200:
        print(f"[DBpedia SPARQL] Query failed for {uri} with status {response.status_code}")
        return []

    results = response.json().get('results', {}).get('bindings', [])
    facts = []
    for result in results:
        prop = result.get('property', {}).get('value', '').split('/')[-1]
        val = result.get('value', {}).get('value', '')
        if prop and val:
            facts.append((prop, val))
    return facts


def query_dbpedia_2hop_facts(uri):
    """
    Query DBpedia 2-hop facts for a given entity URI.
    """
    query = f"""
    SELECT ?p1 ?o1 ?p2 ?o2 WHERE {{
        <{uri}> ?p1 ?o1 .
        OPTIONAL {{ ?o1 ?p2 ?o2 }}
    }}
    LIMIT 50
    """
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(DBPEDIA_SPARQL_ENDPOINT, params={"query": query}, headers=headers)
    if response.status_code != 200:
        print(f"[DBpedia SPARQL] 2-hop Query failed for {uri} with status {response.status_code}")
        return []

    results = response.json().get('results', {}).get('bindings', [])
    facts = []
    for result in results:
        p1 = result.get('p1', {}).get('value', '').split('/')[-1]
        o1 = result.get('o1', {}).get('value', '')
        p2 = result.get('p2', {}).get('value', '').split('/')[-1]
        o2 = result.get('o2', {}).get('value', '')
        if p1 and o1:
            facts.append((p1, o1))
        if p2 and o2:
            facts.append((p2, o2))
    return facts


def get_best_entity_match(surface_form, candidates):
    """
    Choose the best matching candidate entity based on semantic similarity
    """
    surface_emb = embedding_model.encode(surface_form, convert_to_tensor=True)
    best_score = -1
    best_candidate = None
    for c in candidates:
        c_emb = embedding_model.encode(c, convert_to_tensor=True)
        score = util.cos_sim(surface_emb, c_emb).item()
        if score > best_score:
            best_score = score
            best_candidate = c
    return best_candidate

# -----------------------------
# 4. Semantic Filtering
# -----------------------------
def filter_facts_semantically(question, facts, threshold=0.2):
    question_emb = embedding_model.encode(question, convert_to_tensor=True)
    filtered = []

    for prop, val in facts:
        prop_emb = embedding_model.encode(prop, convert_to_tensor=True)
        similarity = util.cos_sim(question_emb, prop_emb).item()

        if similarity >= threshold:
            filtered.append((prop, val))

    return filtered


def convert_facts_to_sentences_auto(entity_label, facts, entity_types=None, context_question=None):
    """
    Converts facts to natural sentences without manual keywords.
    Uses semantic similarity to decide phrasing.
    """
    sentences = []
    entity_label_clean = entity_label.strip().rstrip(".")

    # Optional: encode context question to guide phrasing
    context_emb = None
    if context_question:
        context_emb = embedding_model.encode(context_question, convert_to_tensor=True)

    for prop, val in facts:
        prop_text = prop.strip()
        val_text = val.strip()

        # Base sentence
        sentence = f"{entity_label_clean}'s {prop_text} is {val_text}."

        # Enhance sentence if context provided
        if context_emb:
            # Encode prop + val
            prop_emb = embedding_model.encode(prop_text, convert_to_tensor=True)
            val_emb = embedding_model.encode(val_text, convert_to_tensor=True)

            # Similarity to context
            prop_sim = util.cos_sim(context_emb, prop_emb).item()
            val_sim = util.cos_sim(context_emb, val_emb).item()
            combined_sim = 0.6 * prop_sim + 0.4 * val_sim

            # If highly relevant, rephrase as "had impact/influence"
            if combined_sim > 0.6:
                sentence = f"{entity_label_clean} is related to {val_text} ({prop_text})."

        sentences.append(sentence)

    return sentences



def filter_facts_semantically_and_relevant_auto(question, facts, question_entities=None, top_k_per_entity=5):
    """
    Enhanced semantic filtering:
    - Uses entity-aware similarity
    - Weights value higher for generic properties
    - Ranks facts instead of strict thresholding
    - Keeps top-k most relevant facts per entity
    """
    if not facts:
        return []

    # Encode question
    question_emb = embedding_model.encode(question, convert_to_tensor=True)

    # Encode question entities if provided
    entity_embeddings = []
    if question_entities:
        for ent in question_entities:
            ent_emb = embedding_model.encode(ent, convert_to_tensor=True)
            entity_embeddings.append(ent_emb)

    scored_facts = []

    for prop, val in facts:
        if not prop or not val:
            continue

        # Skip RDF/URL noise
        if prop.lower().startswith(('rdf', 'schema', 'wiki', 'http')):
            continue

        # Encode property and value
        prop_emb = embedding_model.encode(prop, convert_to_tensor=True)
        val_emb = embedding_model.encode(val, convert_to_tensor=True)

        # Base similarity to question
        prop_sim = util.cos_sim(question_emb, prop_emb).item()
        val_sim = util.cos_sim(question_emb, val_emb).item()

        # Adjust weights dynamically for generic properties using semantic similarity
        prop_similarity_to_generic = util.cos_sim(prop_emb, generic_proto_emb).item()
        if prop_similarity_to_generic >= GENERIC_THRESHOLD:
            sim = 0.2 * prop_sim + 0.8 * val_sim
        else:
            sim = 0.6 * prop_sim + 0.4 * val_sim

        # Boost based on entity relevance
        entity_sim_boost = 0
        for ent_emb in entity_embeddings:
            entity_sim_boost = max(entity_sim_boost, util.cos_sim(val_emb, ent_emb).item())
        sim += 0.1 * entity_sim_boost  # small boost for entity-related facts

        scored_facts.append((sim, prop, val))

    # Rank facts by score descending
    scored_facts.sort(reverse=True, key=lambda x: x[0])

    # Return top-k facts
    filtered = [(prop, val) for score, prop, val in scored_facts[:top_k_per_entity]]

    return filtered


def correct_entity_labels(entities_dict):
    corrected = {}
    for eid, label in entities_dict.items():
        aliases = []
        if eid.startswith('Q'):  # Wikidata
            aliases = wikidata_entities(eid) + [label]
        else:  # DBpedia URI
            aliases = [label]
        best_label = get_best_entity_match(label, aliases) if aliases else label
        corrected[eid] = best_label
    return corrected



# -----------------------------
# 5. Main Pipeline (Falcon-only)
# -----------------------------
def process_facts(entities, query_func, two_hop_func, label_func=None, question=None, entity_types=None):
    if entity_types is None:
        entity_types = {}
    facts_raw, facts_2hop, facts_combined, facts_filtered, readable_sentences = {}, {}, {}, {}, {}

    for key, label in entities.items():
        raw = query_func(key)
        hop2 = two_hop_func(key)
        combined = list(set(raw + hop2))
        # Filter out irrelevant or noisy properties and values
        irrelevant_props = ['rdf', 'label', 'filename', 'footer', 'comment', 'thumbnail', 'seeAlso', 'sameAs']
        facts = [
            (p, v) for p, v in combined
            if all(not p.lower().startswith(irr) for irr in irrelevant_props)
               and not v.lower().startswith('http')
               and len(v.strip()) > 1
        ]

        filtered = filter_facts_semantically_and_relevant_auto(
            question,
            facts,
            question_entities=entity_types.get(key, [])
        ) if question else facts
        label_to_use = label_func(key) if label_func else label
        sentences = convert_facts_to_sentences_auto(label_to_use, filtered, entity_types.get(key, []))

        facts_raw[key] = raw
        facts_2hop[key] = hop2
        facts_combined[key] = combined
        facts_filtered[key] = filtered
        readable_sentences[key] = sentences

    return facts_raw, facts_2hop, facts_combined, facts_filtered, readable_sentences


def filter_dbpedia_entities(dbpedia_entities, falcon_labels=None, question=None, threshold=0.5):
    filtered = {}
    falcon_labels = [l.lower() for l in falcon_labels] if falcon_labels else []
    question_emb = embedding_model.encode(question, convert_to_tensor=True) if question else None

    for uri, label in dbpedia_entities.items():
        label_clean = label.strip()
        if len(label_clean) <= 3:
            continue
        if any(label_clean.lower() in wd_label for wd_label in falcon_labels):
            continue
        # Semantic filter: label must be related to question
        if question_emb:
            label_emb = embedding_model.encode(label_clean, convert_to_tensor=True)
            sim = util.cos_sim(question_emb, label_emb).item()
            if sim < threshold:
                continue
        filtered[uri] = label_clean
    return filtered



def process_dbpedia_facts_semantic(dbpedia_entities, question=None, threshold=0.50):

    facts_filtered = {}
    sentences = {}

    # --- Encode question context ---
    if question:
        question_emb = embedding_model.encode(question, convert_to_tensor=True)
    else:
        question_emb = None

    # --- Step 1: Semantic entity filtering ---
    semantic_entities = {}
    for uri, label in dbpedia_entities.items():
        label_clean = label.strip()
        if not label_clean:
            continue

        if question_emb is not None:
            label_emb = embedding_model.encode(label_clean, convert_to_tensor=True)
            sim = util.cos_sim(question_emb, label_emb).item()
            if sim < threshold:
                continue

        semantic_entities[uri] = label_clean

    # --- Step 2: Retrieve and semantically filter facts ---
    for uri, label in semantic_entities.items():
        raw = query_dbpedia_facts(uri)
        hop2 = query_dbpedia_2hop_facts(uri)
        combined = list(set(raw + hop2))

        filtered = []
        for prop, val in combined:
            if not prop or not val:
                continue
            if prop.lower().startswith(('rdf', 'schema', 'wiki', 'http')):
                continue

            prop_emb = embedding_model.encode(prop, convert_to_tensor=True)
            val_emb = embedding_model.encode(val, convert_to_tensor=True)

            prop_sim = util.cos_sim(question_emb, prop_emb).item() if question_emb is not None else 0
            val_sim = util.cos_sim(question_emb, val_emb).item() if question_emb is not None else 0

            sim = 0.6 * prop_sim + 0.4 * val_sim

            if sim >= threshold:
                filtered.append((prop, val))

        facts_filtered[uri] = filtered

        # --- Step 3: Convert filtered facts to natural sentences ---
        entity_label_clean = label.strip().rstrip(".")
        entity_sentences = []
        for prop, val in filtered:
            prop_text = prop.strip()
            val_text = val.strip()

            # Default sentence
            sentence = f"{entity_label_clean}'s {prop_text} is {val_text}."

            # Semantic enhancement if question context is present
            if question_emb is not None:
                prop_emb = embedding_model.encode(prop_text, convert_to_tensor=True)
                val_emb = embedding_model.encode(val_text, convert_to_tensor=True)
                prop_sim = util.cos_sim(question_emb, prop_emb).item()
                val_sim = util.cos_sim(question_emb, val_emb).item()
                combined_sim = 0.6 * prop_sim + 0.4 * val_sim
                if combined_sim > 0.6:
                    sentence = f"{entity_label_clean} is related to {val_text} ({prop_text})."

            entity_sentences.append(sentence)

        sentences[uri] = entity_sentences

    return facts_filtered, sentences


def merge_and_clean_entities(falcon_entities, spotlight_entities):

    # Step 1: Merge (Falcon priority)
    merged = {**spotlight_entities, **falcon_entities}

    # Step 2: Clean overlapping substring entities
    clean = {}
    labels = list(merged.values())

    for uri, label in merged.items():
        label_lower = label.lower()
        # skip short/partial labels if they appear inside a longer one
        if any(
            (label_lower in other.lower() and label_lower != other.lower())
            for other in labels
        ):
            continue
        clean[uri] = label

    return clean

def get_entity_facts(qid, property_filter=None, limit=50):
    """
    Query Wikidata for specific property facts of an entity (optionally filtered by property).
    Returns a list of dicts like:
      {"property": "product", "value": "Windows 11", "pid": "P1056"}
    """
    endpoint = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint)
    sparql.setReturnFormat(JSON)

    filter_clause = f"FILTER(?p = wd:{property_filter})" if property_filter else ""

    query = f"""
    SELECT ?p ?pLabel ?o ?oLabel WHERE {{
      wd:{qid} ?p ?o .
      {filter_clause}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT {limit}
    """

    sparql.setQuery(query)

    try:
        results = sparql.query().convert()
    except Exception as e:
        print(f"⚠️ SPARQL query failed for {qid} ({property_filter}): {e}")
        return []

    facts = []
    for r in results["results"]["bindings"]:
        p_uri = r["p"]["value"]
        p_label = r.get("pLabel", {}).get("value", "")
        o_label = r.get("oLabel", {}).get("value", "")
        pid = p_uri.split("/")[-1]
        facts.append({
            "property": p_label or pid,
            "value": o_label,
            "pid": pid,
        })

    return facts

def run_sparql_query(query, endpoint="https://query.wikidata.org/sparql"):
    """
    Executes a SPARQL query against Wikidata and returns JSON results as a list of dicts.
    Handles rate limits (429) gracefully.
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "EntityLinkingPipeline/1.0 (https://example.org)"
    }
    try:
        response = requests.get(endpoint, params={"query": query}, headers=headers, timeout=30)
        if response.status_code == 429:
            print("[SPARQL] Rate limit hit (429) — retrying after delay...")
            import time; time.sleep(5)
            response = requests.get(endpoint, params={"query": query}, headers=headers, timeout=30)

        response.raise_for_status()
        data = response.json()
        results = []
        for b in data.get("results", {}).get("bindings", []):
            row = {}
            for k, v in b.items():
                row[k] = v.get("value", "")
            results.append(row)
        return results

    except Exception as e:
        print(f"[SPARQL ERROR] {e}")
        return []


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
    def rank_sentences_by_similarity(question, sentences, min_score=0.45):
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
    top_sentences = ranked_expanded[:5]

    reformulated_query = f"{original_text} {' '.join(top_sentences)}"
    reformulated_query = re.sub(r"\s*\.\s*$", "", reformulated_query.strip())
    # if falcon_relations:
    #     reformulated_query += "  🔍 Relation terms incorporated into reformulated query."

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




# -----------------------------
# Clean & Structured Print
# -----------------------------
def print_clean_pipeline_result(result, max_facts_per_entity=5):
    entity_labels = {**result['falcon_qids'], **result['dbpedia_entities']}

    print("\n" + "=" * 50)
    print(f"Original Query: {result['original_query']}")
    print("=" * 50)

    # --- Entities ---
    print("\n[Entities]")
    for eid, label in entity_labels.items():
        source = "Wikidata" if eid in result['falcon_qids'] else "DBpedia"
        print(f"  {source}: {label} ({eid})")

    # --- Merge and deduplicate facts ---
    def normalize_facts(facts_dict):
        merged = {}
        for eid, facts in facts_dict.items():
            unique = set((prop.lower(), val.lower()) for prop, val in facts)
            merged[eid] = [(prop.capitalize(), val.capitalize()) for prop, val in unique]
        return merged

    merged_wikidata = normalize_facts(result['wikidata_facts_filtered'])
    merged_dbpedia = normalize_facts(result['dbpedia_facts_filtered'])

    # --- Filtered Facts ---
    if any(merged_wikidata.values()) or any(merged_dbpedia.values()):
        print("\n[Filtered Facts]")
        for eid, facts in merged_wikidata.items():
            if facts:
                print(f"  {result['falcon_qids'][eid]} (Wikidata):")
                for prop, val in facts[:max_facts_per_entity]:
                    print(f"    - {prop}: {val}")
        for eid, facts in merged_dbpedia.items():
            if facts:
                print(f"  {result['dbpedia_entities'][eid]} (DBpedia):")
                for prop, val in facts[:max_facts_per_entity]:
                    print(f"    - {prop}: {val}")

    # --- Natural Language Summary ---
    if any(result['natural_language_summary'].values()):
        print("\n[Natural Language Summary]")
        for eid, sentences in result['natural_language_summary'].items():
            if sentences:
                label = entity_labels.get(eid, eid)
                unique_sentences = sorted(set(sentences))
                for sentence in unique_sentences[:max_facts_per_entity]:
                    print(f"  - {sentence}")

    # --- Reformulated Query (concise) ---
    top_sentences = []
    for sentences in result['natural_language_summary'].values():
        top_sentences.extend(sentences[:max_facts_per_entity])
    reformulated_query = f"{result['original_query']} {' '.join(top_sentences)}"

    print("\n[Reformulated Query]")
    print(f"  {reformulated_query}")
    print("=" * 50)


def display_full_pipeline_result(result, max_facts_per_entity=10, show_scores=True):
    """
    Deep diagnostic display.
    Shows detailed provenance, property IDs, scores, SPARQL URLs,
    and how relations (e.g. 'products') are applied to entities.
    """

    # === Core Display Utilities ===
    def print_entities():
        print("\n[Entities & Linking]")
        all_entities = {**result.get('falcon_qids', {}), **result.get('dbpedia_entities', {})}
        entity_types = result.get('entity_types', {})
        for eid, label in all_entities.items():
            src = "Wikidata" if eid in result.get('falcon_qids', {}) else "DBpedia"
            types = ", ".join(entity_types.get(eid, []))
            url = f"https://www.wikidata.org/wiki/{eid}" if src == "Wikidata" else f"http://dbpedia.org/resource/{eid}"
            print(f"  - {label} ({src}, {eid})")
            if types:
                print(f"      Types: {types}")
            print(f"      URL: {url}")

    def print_facts_section(facts_dict, entities_dict, title, source="wikidata"):
        print(f"\n[{title}]")
        for eid, facts in facts_dict.items():
            if not facts:
                continue
            elabel = entities_dict.get(eid, eid)
            print(f"→ {elabel} ({eid})")
            for f in facts[:max_facts_per_entity]:
                if isinstance(f, dict):
                    prop = f.get("property") or f.get("p")
                    val = f.get("value") or f.get("v")
                    pid = f.get("pid")
                    score = f.get("score")
                    prov = f.get("provenance", "")
                elif isinstance(f, (list, tuple)) and len(f) >= 2:
                    prop, val = f[:2]
                    pid = f[2] if len(f) > 2 else None
                    score = f[3] if len(f) > 3 else None
                    prov = f[4] if len(f) > 4 else ""
                else:
                    continue

                pid_str = f" ({pid})" if pid else ""
                prov_str = f" ← {prov}" if prov else ""
                score_str = f"  [score={score:.3f}]" if show_scores and isinstance(score, (int, float)) else ""
                prop_url = f"https://www.wikidata.org/wiki/Property:{pid}" if pid and source == "wikidata" else ""
                val_url = f"https://www.wikidata.org/wiki/{val}" if isinstance(val, str) and val.startswith("Q") else ""
                val_display = f"{val} ({val_url})" if val_url else val

                if prop_url:
                    print(f"    • {prop}{pid_str}: {val_display}{score_str}{prov_str}")
                    print(f"       → {prop_url}")
                else:
                    print(f"    • {prop}{pid_str}: {val_display}{score_str}{prov_str}")

    def print_relation_diagnostics():
        print("\n[Relations Identified]")
        rels = result.get("falcon_relations", {})
        if not rels:
            print("  (none)")
            return
        for rid, label in rels.items():
            wikidata_url = f"https://www.wikidata.org/wiki/Property:{rid}" if rid.startswith("P") else f"http://dbpedia.org/ontology/{rid}"
            print(f"  - {label} ({rid})")
            print(f"      URL: {wikidata_url}")

    # 🔍 NEW: Show which entities have facts for each relation
    def print_relation_entity_mapping():
        print("\n[Relations ↔ Entities Mapping]")
        rels = result.get("falcon_relations", {})
        if not rels:
            print("  (none)")
            return
        found_any = False
        for rid, rlabel in rels.items():
            matching_entities = []
            for eid, facts in result.get('wikidata_facts_combined', {}).items():
                if any(
                        (
                                isinstance(f, dict)
                                and any(rlabel.lower() in str(v).lower() or rid in str(v) for v in f.values())
                        ) or (
                                isinstance(f, (list, tuple))
                                and any(rlabel.lower() in str(x).lower() or rid in str(x) for x in f)
                        ) or (
                                isinstance(f, str)
                                and (rlabel.lower() in f.lower() or rid in f)
                        )
                        for f in facts
                ):
                    matching_entities.append(result['falcon_qids'].get(eid, eid))
            if matching_entities:
                found_any = True
                print(f"  - {rlabel} ({rid}) → {', '.join(matching_entities)}")
            else:
                print(f"  - {rlabel} ({rid}) → no matching facts found")
        if not found_any:
            print("  (no relation-linked entities)")

    # 🔍 NEW: Print facts that correspond directly to relations (like 'products')
    def print_relation_facts():
        print("\n[Relation-driven Facts]")
        rels = result.get("falcon_relations", {})
        if not rels:
            print("  (none)")
            return
        for rid, rlabel in rels.items():
            print(f"\n→ Relation: {rlabel} ({rid})")
            found = False
            for eid, facts in result.get('wikidata_facts_combined', {}).items():
                elabel = result['falcon_qids'].get(eid, eid)
                for f in facts:
                    prop = f[0] if isinstance(f, (list, tuple)) else f.get("property")
                    val = f[1] if isinstance(f, (list, tuple)) else f.get("value")
                    if rlabel.lower() in str(prop).lower() or rid in str(prop):
                        found = True
                        print(f"   • {elabel} → {rlabel}: {val}")
            if not found:
                print("   (no matching facts for this relation)")

    def trace_filtered_facts():
        print("\n[Trace: Filtered Fact Origins]")
        filtered = result.get("wikidata_facts_filtered", {})
        if not filtered:
            print("  (none)")
            return
        for eid, facts in filtered.items():
            label = result['falcon_qids'].get(eid, eid)
            for fact in facts:
                prop, val = fact[:2]
                origin = []
                if (prop, val) in result['wikidata_facts'].get(eid, []):
                    origin.append("1-hop")
                if (prop, val) in result['wikidata_facts_2hop'].get(eid, []):
                    origin.append("2-hop")
                if (prop, val) in result['wikidata_facts_combined'].get(eid, []):
                    origin.append("combined")
                origin_str = "/".join(origin) or "unknown"
                print(f"  {label} • [{prop}] {val}  ← came from {origin_str}")

    def print_summary():
        print("\n[Natural Language Summary]")
        summary = result.get('natural_language_summary', {})
        for eid, sentences in summary.items():
            label = result['falcon_qids'].get(eid, result['dbpedia_entities'].get(eid, eid))
            print(f"→ {label}")
            for s in sentences:
                relation_mark = ""
                for rel_label in result.get("falcon_relations", {}).values():
                    if rel_label.lower() in s.lower():
                        relation_mark = "  🔍 [relation-based]"
                        break
                print(f"   • {s}{relation_mark}")

    def print_reformulated_query():
        print("\n[Reformulated Query]")
        rq = result.get('reformulated_query', '')
        if any(rel.lower() in rq.lower() for rel in result.get("falcon_relations", {}).values()):
            print(f"  {rq}")
            print("  🔍 Relation terms incorporated into reformulated query.")
        else:
            print(f"  {rq}")

    # === Display Order ===
    print("=" * 100)
    print(f"Original Query:\n  {result.get('original_query', '')}")
    print("=" * 100)

    print_entities()
    print_relation_diagnostics()
    print_relation_entity_mapping()   # 🔍 NEW
    print_relation_facts()            # 🔍 NEW

    print_facts_section(result.get('wikidata_facts', {}), result.get('falcon_qids', {}), "Wikidata Facts (1-hop)", "wikidata")
    print_facts_section(result.get('wikidata_facts_2hop', {}), result.get('falcon_qids', {}), "Wikidata Facts (2-hop)", "wikidata")
    print_facts_section(result.get('wikidata_facts_combined', {}), result.get('falcon_qids', {}), "Wikidata Facts (Combined)", "wikidata")
    print_facts_section(result.get('wikidata_facts_filtered', {}), result.get('falcon_qids', {}), "Wikidata Facts (Filtered)", "wikidata")

    print_facts_section(result.get('dbpedia_facts_raw', {}), result.get('dbpedia_entities', {}), "DBpedia Facts (1-hop)", "dbpedia")
    print_facts_section(result.get('dbpedia_facts_2hop', {}), result.get('dbpedia_entities', {}), "DBpedia Facts (2-hop)", "dbpedia")
    print_facts_section(result.get('dbpedia_facts_combined', {}), result.get('dbpedia_entities', {}), "DBpedia Facts (Combined)", "dbpedia")
    print_facts_section(result.get('dbpedia_facts_filtered', {}), result.get('dbpedia_entities', {}), "DBpedia Facts (Filtered)", "dbpedia")

    trace_filtered_facts()
    print_summary()
    print_reformulated_query()

    print("=" * 100)


# -----------------------------
# Example Usage
# # -----------------------------
# prompt = "water flux meaning"
# result = enrich_query_with_entities_and_facts(prompt)
# # print_clean_pipeline_result(result)
#
# display_full_pipeline_result(result,1)

