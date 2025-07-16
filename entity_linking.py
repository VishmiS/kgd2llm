import requests
import spacy
from pprint import pprint
from sentence_transformers import SentenceTransformer, util

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load SentenceTransformer model for semantic similarity
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"


def falcon_entity_linking(text):
    url = "https://labs.tib.eu/falcon/falcon2/api?mode=long"
    payload = {"text": text}
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        entities = result.get("entities_wikidata", [])
        return {entity['URI'].split('/')[-1]: entity['surface form']
                for entity in entities if 'URI' in entity}
    return {}


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


def spacy_entities(text):
    doc = nlp(text)
    return list(set(ent.text for ent in doc.ents))


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


def enrich_query_with_aliases_and_facts(original_text):
    print("\n" + "=" * 50)
    print(f"[Pipeline] Original Query: {original_text}")
    print("=" * 50)

    spacy_ents = spacy_entities(original_text)
    falcon_qids = {}
    for ent in spacy_ents:
        ent_qids = falcon_entity_linking(ent)
        falcon_qids.update(ent_qids)

    qid_to_aliases = {qid: wikidata_aliases(qid) for qid in falcon_qids}
    qid_to_facts_raw = {qid: query_wikidata_facts(qid) for qid in falcon_qids}
    qid_to_facts_2hop = {qid: query_2hop_facts(qid) for qid in falcon_qids}

    qid_to_facts_combined = {}
    for qid in falcon_qids:
        combined = qid_to_facts_raw.get(qid, []) + qid_to_facts_2hop.get(qid, [])
        qid_to_facts_combined[qid] = list(set(combined))

    qid_to_facts_filtered = {
        qid: filter_facts_semantically(original_text, facts)
        for qid, facts in qid_to_facts_combined.items()
    }

    expanded_forms = set()
    for aliases in qid_to_aliases.values():
        expanded_forms.update(aliases)

    readable_sentences = {}
    for qid, facts in qid_to_facts_filtered.items():
        label = falcon_qids[qid]
        readable_sentences[qid] = convert_facts_to_sentences(label, facts)

    # Reformulated query: include aliases and the readable sentences
    reformulated_query = (
        original_text + " "
        + " ".join(expanded_forms) + " "
        + " ".join(
            sentence
            for sentences in readable_sentences.values()
            for sentence in sentences
        )
    )

    result = {
        "original_query": original_text,
        "spaCy_entities": spacy_ents,
        "falcon_qids": falcon_qids,
        "wikidata_aliases": qid_to_aliases,
        "wikidata_facts": qid_to_facts_raw,
        "wikidata_facts_2hop": qid_to_facts_2hop,
        "wikidata_facts_combined": qid_to_facts_combined,
        "wikidata_facts_filtered": qid_to_facts_filtered,
        "reformulated_query": reformulated_query.strip(),
        "natural_language_summary": readable_sentences,
    }

    print("\n[Pipeline] Final Output:")
    pprint(result)
    return result


# Example usage
prompt = "Who is the CEO of Microsoft and where is their headquarters located?"
result = enrich_query_with_aliases_and_facts(prompt)

# Show readable sentences
print("\n=== Natural Language Summary ===")
for qid, sentences in result["natural_language_summary"].items():
    print(f"\n[Entity: {result['falcon_qids'][qid]}]")
    for sentence in sentences:
        print(f"- {sentence}")

print("\n=== Reformulated Query ===")
print(result["reformulated_query"])
