import requests
from sentence_transformers import SentenceTransformer, util
import time
import re
import json
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
from entity_linking.view_results import display_full_pipeline_result
from entity_linking.decompose_phrase import resolve_composite_entity

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
def dbpedia_entity_linking(text, confidence=0.2, support=20):
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
        p = result.get('property', {}).get('value', '')
        v = result.get('value', {}).get('value', '')

        prop = p.split('/')[-1] if '/' in p else p
        val = v.strip()

        if prop and val:
            facts.append((prop, val))

    return facts


def is_technical_id(value):
    """Check if a value is a technical ID/code that should be filtered out"""
    if not value:
        return True

    value_str = str(value).strip().lower()

    # Common technical ID patterns to reject
    technical_patterns = [
        r'^[nq][m]\d+',  # nm1234, qm1234 (IMDb, etc.)
        r'^\d+$',  # Pure numbers
        r'^[a-z]{1,3}\d+',  # Short codes with letters+numbers
        r'^tt\d+',  # IMDb title IDs
        r'^ch\d+',  # Character IDs
    ]

    for pattern in technical_patterns:
        if re.match(pattern, value_str):
            return True

    # Reject common ID labels
    id_indicators = ['imdb', 'isbn', 'issn', 'doi', 'id:', 'code:', 'number:']
    if any(indicator in value_str for indicator in id_indicators):
        return True

    # Reject very short values with digits
    if len(value_str) <= 4 and any(c.isdigit() for c in value_str):
        return True

    return False

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
    results = response.json().get('results', {}).get('bindings', [])
    facts = []
    for result in results:
        prop_uri = result.get('property', {}).get('value', '')
        val = result.get('value', {}).get('value', '')
        if not prop_uri or not val:
            continue

        # Normalize property namespace
        if "ontology/" in prop_uri:
            p = "dbo:" + prop_uri.split("ontology/")[-1]
        elif "property/" in prop_uri:
            p = "dbp:" + prop_uri.split("property/")[-1]
        else:
            p = prop_uri.split("/")[-1]

        # ✅ Include both dbo: and dbp: facts
        if p.startswith("dbo:") or p.startswith("dbp:"):
            facts.append((p, val))

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


def filter_facts_semantically_and_relevant_auto(
    question: str,
    facts: list,
    question_entities: list = None,
    top_k_per_entity: int = 5,
    threshold: float = 0.2
):
    """
    Filters and ranks (property, value) facts based on semantic similarity
    to a given natural language question, with synonym expansion and
    contextual relevance boosting.
    """
    if not facts:
        return []

    # --- 1️⃣ Expand the question with semantic synonyms ---
    synonyms = {
        "where": ["place", "location", "city", "birthplace", "country", "origin"],
        "who": ["person", "individual", "human", "biography"],
        "when": ["date", "time", "year", "birthdate", "foundation"],
        "from": ["born in", "origin", "place of birth", "birthplace"]
    }
    expanded = [question]
    for key, vals in synonyms.items():
        if key in question.lower():
            expanded.extend(vals)
    expanded_question = " ".join(expanded)

    # --- 2️⃣ Encode question ---
    question_emb = embedding_model.encode(expanded_question, convert_to_tensor=True)

    # --- 3️⃣ Encode entities if provided ---
    entity_embeddings = []
    if question_entities:
        for ent in question_entities:
            ent_emb = embedding_model.encode(ent, convert_to_tensor=True)
            entity_embeddings.append(ent_emb)

    scored_facts = []

    # --- 4️⃣ Score each fact ---
    for prop, val in facts:
        if not prop or not val:
            continue

        # Skip RDF/URL noise
        if prop.lower().startswith(('rdf', 'schema', 'wiki', 'http')):
            continue

        # Encode property and value
        prop_emb = embedding_model.encode(prop, convert_to_tensor=True)
        val_emb = embedding_model.encode(val, convert_to_tensor=True)

        # Compute base similarities
        prop_sim = util.cos_sim(question_emb, prop_emb).item()
        val_sim = util.cos_sim(question_emb, val_emb).item()

        # --- 5️⃣ Adjust dynamic weighting ---
        prop_similarity_to_generic = 0.0
        if 'generic_proto_emb' in globals():
            prop_similarity_to_generic = util.cos_sim(prop_emb, generic_proto_emb).item()

        if prop_similarity_to_generic >= 0.6:
            sim = 0.2 * prop_sim + 0.8 * val_sim
        else:
            sim = 0.6 * prop_sim + 0.4 * val_sim

        # --- 6️⃣ Add entity relevance boost ---
        entity_sim_boost = 0
        for ent_emb in entity_embeddings:
            entity_sim_boost = max(entity_sim_boost, util.cos_sim(val_emb, ent_emb).item())
        sim += 0.1 * entity_sim_boost

        # --- 7️⃣ Contextual keyword boosting ---
        keywords = ["birth", "born", "origin", "place", "hometown", "city", "country", ]
        if any(k in prop.lower() for k in keywords):
            sim += 0.15
        if "where" in question.lower():
            sim += 0.1  # stronger boost for location questions
        if "when" in question.lower() and any(
                k in prop.lower()
                for k in ["date", "year", "time", "birthdate", "first performance", "release", "opening"]
        ):
            sim += 0.1

        scored_facts.append((sim, prop, val))

    # --- 8️⃣ Rank and threshold ---
    scored_facts.sort(reverse=True, key=lambda x: x[0])
    filtered = [(prop, val) for sim, prop, val in scored_facts if sim >= threshold]

    # --- 9️⃣ Limit per entity ---
    return filtered[:top_k_per_entity]


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

    from concurrent.futures import ThreadPoolExecutor

    def process_entity(key_label):
        key, label = key_label
        raw = query_func(key)
        hop2 = two_hop_func(key)
        # Combine the facts early (you can change merge logic if needed)
        combined = raw + hop2 if hop2 else raw
        return key, label, raw, hop2, combined

    facts_raw, facts_2hop, facts_combined, facts_filtered, readable_sentences = {}, {}, {}, {}, {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(process_entity, entities.items())

    for key, label, raw, hop2, combined in results:
        # Filter out irrelevant or noisy properties and values
        irrelevant_props = ['rdf', 'label', 'filename', 'footer', 'comment', 'thumbnail', 'seeAlso', 'sameAs']
        facts = [
            (p, v) for p, v in combined
            if all(not p.lower().startswith(irr) for irr in irrelevant_props)
               and not v.lower().startswith('http')
               and len(v.strip()) > 1
               and not is_technical_id(v)
        ]

        filtered = (
            filter_facts_semantically_and_relevant_auto(
                question,
                facts,
                question_entities=entity_types.get(key, [])
            ) if question else facts
        )

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


def run_sparql_query(query, endpoint="https://query.wikidata.org/sparql", max_retries=3, base_delay=5):
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "EntityLinkingPipeline/1.0 (https://example.org)"
    }

    for attempt in range(max_retries):
        try:
            # Calculate delay with exponential backoff (except for first attempt)
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))  # 5, 10, 20 seconds
                print(f"[SPARQL] Waiting {delay} seconds before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)

            response = requests.get(endpoint, params={"query": query}, headers=headers, timeout=30)

            if response.status_code == 200:
                data = response.json()
                results = []
                for b in data.get("results", {}).get("bindings", []):
                    row = {}
                    for k, v in b.items():
                        row[k] = v.get("value", "")
                    results.append(row)
                return results

            elif response.status_code == 429:
                print(f"[SPARQL] Rate limit hit (429) on attempt {attempt + 1}/{max_retries}")
                if attempt == max_retries - 1:
                    print("[SPARQL] Max retries reached for rate limiting")
                    return []
                continue  # Continue to next retry

            else:
                print(f"[SPARQL] HTTP error {response.status_code} on attempt {attempt + 1}")
                response.raise_for_status()  # This will raise an exception for other HTTP errors

        except requests.exceptions.Timeout:
            print(f"[SPARQL] Timeout on attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                return []

        except requests.exceptions.RequestException as e:
            print(f"[SPARQL] Request error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                return []

        except Exception as e:
            print(f"[SPARQL] Unexpected error on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt == max_retries - 1:
                return []

    return []  # Return empty list if all retries fail


# Add these to entity_linking_utils.py

def is_likely_property(term, context_question=""):
    """Check if a term is likely a property rather than an entity with aggressive filtering"""
    if not term:
        return False

    term_lower = str(term).lower()
    question_lower = context_question.lower() if context_question else ""

    # Property words that should ALWAYS be filtered out
    strong_property_indicators = [
        "die", "death", "died", "dead",
        "born", "birth", "birthplace",
        "location", "place", "where",
        "date", "year", "when", "time",
        "who", "person", "what", "which",
        "how", "many", "much"
    ]

    # If it's exactly one of these property words, filter it out
    if term_lower in strong_property_indicators:
        print(f"[PROPERTY FILTER] Filtering '{term}' as strong property indicator")
        return True

    # Check for Wikidata property format (P followed by numbers)
    if re.match(r'^P\d+$', str(term)):
        print(f"[PROPERTY FILTER] Filtering '{term}' as Wikidata property")
        return True

    # Check for DBpedia property format (ontology/property patterns)
    if any(pattern in str(term) for pattern in ["ontology/", "property/", "/prop/"]):
        print(f"[PROPERTY FILTER] Filtering '{term}' as DBpedia property")
        return True

    # If it's a single word and appears in question as a verb/question word, filter it
    if len(term_lower.split()) == 1:
        question_words = question_lower.split()
        if term_lower in question_words:
            # Check if it's being used as a question word or verb
            if term_lower in ["where", "when", "who", "what", "which", "how"]:
                print(f"[PROPERTY FILTER] Filtering '{term}' as question word")
                return True
            # Check if it's a verb in the question context
            if term_lower in ["die", "died", "born", "birth"]:
                print(f"[PROPERTY FILTER] Filtering '{term}' as verb in question")
                return True

    return False


def clean_relations(rel_dict, original_text=""):
    """Clean and filter relation dictionaries"""
    clean_map = {}
    if isinstance(rel_dict, dict):
        for k, v in rel_dict.items():
            if not k or not v:
                continue
            # Enhanced property detection
            if isinstance(k, str):
                # Accept Wikidata properties (P followed by numbers)
                if k.startswith("P") and k[1:].isdigit():
                    if isinstance(v, str) and len(v.strip()) > 2:
                        clean_map[k] = v.strip()
                # Accept DBpedia properties with ontology indicators
                elif any(indicator in k.lower() for indicator in ["ontology", "property", "/prop/"]):
                    if isinstance(v, str) and len(v.strip()) > 2:
                        clean_map[k] = v.strip()
                # Also accept property-like labels
                elif is_likely_property(v, original_text):
                    if isinstance(v, str) and len(v.strip()) > 2:
                        clean_map[k] = v.strip()

    elif isinstance(rel_dict, list):
        for rel in rel_dict:
            if isinstance(rel, dict):
                key = rel.get("relation") or rel.get("uri") or rel.get("id")
                label = rel.get("label") or rel.get("surface_form") or rel.get("text")
                if key and label and len(label.strip()) > 2:
                    # Enhanced filtering for property-like relations
                    if (key.startswith("P") and key[1:].isdigit()) or is_likely_property(label, original_text):
                        clean_map[key] = label.strip()
    return clean_map


def has_wrong_entity_types(falcon_qids, entity_types):
    """Check if we have obviously wrong entity types for the question"""
    if not falcon_qids:
        return False

    # Check if any entities are news articles, websites, or other wrong types for person/location questions
    wrong_types_indicators = ['news article', 'website', 'web page', 'article', 'media']
    for qid in falcon_qids:
        types = entity_types.get(qid, [])
        type_str = ' '.join(types).lower()
        if any(wrong_type in type_str for wrong_type in wrong_types_indicators):
            return True
    return False


def rank_sentences_by_similarity(question, sentences, min_score=0.65, embedding_model=None):
    """Rank sentences by semantic similarity to question"""
    if not embedding_model:
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    q_emb = embedding_model.encode(question, convert_to_tensor=True)
    scored = [(util.cos_sim(q_emb, embedding_model.encode(s, convert_to_tensor=True)).item(), s)
              for s in sentences]
    return [s for score, s in sorted(scored, reverse=True) if score >= min_score]


def is_valid_answer(answer):
    """Check if answer is valid (not URL, not coordinates, reasonable length)"""
    if len(answer) < 2:
        return False

    # Reject URLs
    if re.match(r'https?://', answer) or '.' in answer and (
            'www.' in answer or '.com' in answer or '.org' in answer):
        return False

    # Reject coordinates
    if 'Point(' in answer:
        return False

    # Reject pure numbers
    if re.match(r'^\d+$', answer):
        return False

    # Reject decimal numbers (coordinates)
    if re.match(r'^-?\d+\.\d+$', answer):
        return False

    # Reject answers that are too long
    if len(answer.split()) > 5:
        return False

    # Reject answers that look like timestamps
    if re.match(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', answer):
        return False

    return True


def extract_main_entity_from_question(question_text):
    """Extract the main entity from a question by removing question words and verbs"""
    question_lower = question_text.lower().strip('?')

    # Remove common question patterns
    patterns_to_remove = [
        r'^where did ',
        r'^where was ',
        r'^where is ',
        r'^where are ',
        r'^when did ',
        r'^when was ',
        r'^who did ',
        r'^who was ',
        r'^what did ',
        r'^what was ',
        r'^how did ',
        r'^how was ',
        r'\?$'
    ]

    cleaned = question_lower
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned)

    # Remove common verbs and question words
    stop_words = ['go to', 'study at', 'work at', 'live in', 'born in', 'die in', 'marry']
    for stop_word in stop_words:
        cleaned = cleaned.replace(stop_word, '')

    # Clean up extra spaces and return
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # If we have something reasonable, return it
    if len(cleaned.split()) >= 1 and len(cleaned) > 2:
        return cleaned.title()

    return None