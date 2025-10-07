import requests
from pprint import pprint
from sentence_transformers import SentenceTransformer, util

# Load SentenceTransformer model for semantic similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
DBPEDIA_SPARQL_ENDPOINT = "https://dbpedia.org/sparql"


# -----------------------------
# 1. Falcon Entity Linking
# -----------------------------
def falcon_entity_linking(text):
    """
    Use Falcon 2.0 API to detect and link entities (Wikidata & DBpedia)
    from the input text.
    """
    url = "https://labs.tib.eu/falcon/falcon2/api?mode=long"
    payload = {"text": text}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        entities = result.get("entities_wikidata", [])
        return {entity['URI'].split('/')[-1]: entity['surface form']
                for entity in entities if 'URI' in entity}
    else:
        print(f"[Falcon] Error: {response.status_code}")
        return {}


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
def wikidata_aliases(qid):
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    response = requests.get(url)
    if response.status_code != 200:
        return []
    data = response.json()
    entity_data = data['entities'].get(qid, {})
    aliases = []

    labels = entity_data.get('labels', {})
    if 'en' in labels:
        aliases.append(labels['en']['value'])

    alias_data = entity_data.get('aliases', {})
    if 'en' in alias_data:
        aliases.extend([alias['value'] for alias in alias_data['en']])

    return list(set(aliases))


def query_wikidata_facts(qid):
    query = f"""
    SELECT ?propertyLabel ?valueLabel WHERE {{
      wd:{qid} ?p ?statement .
      ?property wikibase:directClaim ?p .
      OPTIONAL {{ ?statement rdfs:label ?value . FILTER (lang(?value) = "en") }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 50
    """
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(WIKIDATA_SPARQL_ENDPOINT, params={"query": query}, headers=headers)
    if response.status_code != 200:
        print(f"[SPARQL] Query failed for {qid} with status {response.status_code}")
        return []

    results = response.json().get('results', {}).get('bindings', [])
    facts = []
    for result in results:
        prop = result.get('propertyLabel', {}).get('value')
        val = result.get('valueLabel', {}).get('value')
        if prop and val:
            facts.append((prop, val))
    return facts


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


# -----------------------------
# 4. Semantic Filtering
# -----------------------------
def filter_facts_semantically(question, facts, threshold=0.4):
    question_emb = embedding_model.encode(question, convert_to_tensor=True)
    filtered = []

    for prop, val in facts:
        prop_emb = embedding_model.encode(prop, convert_to_tensor=True)
        similarity = util.cos_sim(question_emb, prop_emb).item()

        if similarity >= threshold:
            filtered.append((prop, val))

    return filtered


def convert_facts_to_sentences(qid_label, facts):
    sentences = []
    for prop, val in facts:
        sentences.append(f"{qid_label} has {prop} as {val}.")
    return sentences


# -----------------------------
# 5. Main Pipeline (Falcon-only)
# -----------------------------
def process_facts(entities, query_func, two_hop_func, label_func=None, question=None):
    """
    Generic function to retrieve, combine, filter, and convert facts to readable sentences.

    Args:
        entities: dict, {id_or_uri: label}
        query_func: function to retrieve 1-hop facts
        two_hop_func: function to retrieve 2-hop facts
        label_func: optional function to get label (for Wikidata aliases)
        question: original query for semantic filtering

    Returns:
        facts_raw, facts_2hop, facts_combined, facts_filtered, readable_sentences
    """
    facts_raw, facts_2hop, facts_combined, facts_filtered, readable_sentences = {}, {}, {}, {}, {}

    for key, label in entities.items():
        raw = query_func(key)
        hop2 = two_hop_func(key)
        combined = list(set(raw + hop2))
        filtered = filter_facts_semantically(question, combined) if question else combined
        label_to_use = label_func(key) if label_func else label
        sentences = convert_facts_to_sentences(label_to_use, filtered)

        facts_raw[key] = raw
        facts_2hop[key] = hop2
        facts_combined[key] = combined
        facts_filtered[key] = filtered
        readable_sentences[key] = sentences

    return facts_raw, facts_2hop, facts_combined, facts_filtered, readable_sentences


def enrich_query_with_aliases_and_facts(original_text):
    # ------------------- Entity Linking -------------------
    falcon_qids = falcon_entity_linking(original_text)
    dbpedia_entities = dbpedia_entity_linking(original_text)

    if not falcon_qids:
        print("[Pipeline] No Falcon entities found, using DBpedia only.")
    if not dbpedia_entities:
        print("[Pipeline] No DBpedia entities found.")

    # ------------------- Wikidata Processing -------------------
    qid_to_aliases = {qid: wikidata_aliases(qid) for qid in falcon_qids}
    (qid_facts_raw, qid_facts_2hop, qid_facts_combined,
     qid_facts_filtered, readable_sentences) = process_facts(
        falcon_qids,
        query_wikidata_facts,
        query_2hop_facts,
        label_func=lambda qid: falcon_qids[qid],
        question=original_text
    )

    # ------------------- DBpedia Processing -------------------
    (dbpedia_facts_raw, dbpedia_facts_2hop, dbpedia_facts_combined,
     dbpedia_facts_filtered, dbpedia_readable_sentences) = process_facts(
        dbpedia_entities,
        query_dbpedia_facts,
        query_dbpedia_2hop_facts,
        question=original_text
    )

    # ------------------- Expanded Forms -------------------
    expanded_forms = set()
    for aliases in qid_to_aliases.values():
        expanded_forms.update(aliases)

    # ------------------- Reformulated Query -------------------
    reformulated_query = " ".join([
        original_text,
        *expanded_forms,
        *[sentence for sentences in readable_sentences.values() for sentence in sentences],
        *[sentence for sentences in dbpedia_readable_sentences.values() for sentence in sentences]
    ])


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

    # --- Wikidata Aliases ---
    if any(result['wikidata_aliases'].values()):
        print("\n[Wikidata Aliases]")
        for qid, aliases in result['wikidata_aliases'].items():
            if aliases:
                print(f"  {result['falcon_qids'][qid]} ({qid}): {', '.join(aliases)}")

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



# -----------------------------
# Example Usage
# -----------------------------
# prompt = "occupation of Barack Obama?"
# result = enrich_query_with_aliases_and_facts(prompt)
# print_clean_pipeline_result(result)


