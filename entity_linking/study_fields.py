
from sentence_transformers import util

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Make sure embedding_model is available - add this if it's not already defined
try:
    from your_main_file import embedding_model  # Adjust import path as needed
except ImportError:
    # Fallback: define it here if not available
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_property_value(entity_qid: str, property_id: str):
    """
    Query Wikidata for the value(s) of a given property for a given entity.
    Returns list of (value_id, value_label).
    """
    import requests

    query = f"""
    SELECT ?value ?valueLabel WHERE {{
      wd:{entity_qid} wdt:{property_id} ?value .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 10
    """

    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "EntityLinkerBot/1.0 (https://example.org)"}

    try:
        r = requests.get(url, params={"query": query, "format": "json"}, headers=headers, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", {}).get("bindings", [])
        return [(res["value"]["value"].split("/")[-1], res["valueLabel"]["value"]) for res in results]
    except Exception as e:
        print(f"[SPARQL value lookup failed] for {entity_qid}, {property_id}: {e}")
        return []

def get_field_from_occupation_via_wikidata(occupation_qid: str):
    """
    Use Wikidata to find the academic field associated with an occupation.
    """
    query = """
    SELECT ?field ?fieldLabel WHERE {
      {
        # Check if occupation has a 'field of this occupation' (P425)
        wd:%s wdt:P425 ?field .
      } UNION {
        # Check if occupation is a subclass of something with a field
        wd:%s wdt:P279+ ?parent .
        ?parent wdt:P425 ?field .
      } UNION {
        # Check for 'main subject' (P921) of the occupation
        wd:%s wdt:P921 ?field .
      }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
    }
    LIMIT 5
    """

    try:
        from entity_linking.entity_linking_utils import run_sparql_query
        results = run_sparql_query(query % (occupation_qid, occupation_qid, occupation_qid))
        fields = []
        for res in results:
            field_qid = res.get("field", {}).get("value", "").split("/")[-1]
            field_label = res.get("fieldLabel", {}).get("value", "")
            if field_label:
                fields.append((field_qid, field_label))
        return fields
    except Exception as e:
        print(f"[WIKIDATA FIELD LOOKUP] Failed for {occupation_qid}: {e}")
        return []


def find_semantically_similar_fields(occupation_label: str, top_k: int = 3):
    """
    Use semantic similarity to find the closest academic fields.
    """
    # Common academic fields
    academic_fields = [
        "physics", "mathematics", "biology", "chemistry", "computer science",
        "engineering", "medicine", "astronomy", "geology", "psychology",
        "economics", "philosophy", "history", "literature", "art", "music",
        "political science", "sociology", "anthropology", "linguistics"
    ]

    # Encode occupation and fields
    occ_emb = embedding_model.encode(occupation_label, convert_to_tensor=True)
    field_embs = embedding_model.encode(academic_fields, convert_to_tensor=True)

    # Calculate similarities
    similarities = util.cos_sim(occ_emb, field_embs)[0]

    # Get top matches
    top_indices = similarities.topk(top_k).indices
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.4:  # similarity threshold
            results.append(("FIELD", academic_fields[idx]))  # Using "FIELD" as placeholder QID

    return results


def get_academic_field_for_person(entity_qid: str, entity_label: str):
    """
    Comprehensive approach to find academic field using multiple strategies.
    """
    field_results = []

    # Strategy 1: Direct field properties
    direct_field_props = ["P101", "P69"]  # field of work, educated at
    for pid in direct_field_props:
        values = get_property_value(entity_qid, pid)
        if values:
            field_results.extend(values)

    if field_results:
        return field_results

    # Strategy 2: Occupation-based field inference via Wikidata
    occupation_values = get_property_value(entity_qid, "P106")
    for occ_qid, occ_label in occupation_values:
        print(f"[COMPOSITE] Checking field for occupation: {occ_label} ({occ_qid})")

        # Try Wikidata relationships first
        wikidata_fields = get_field_from_occupation_via_wikidata(occ_qid)
        if wikidata_fields:
            field_results.extend(wikidata_fields)
            print(f"[COMPOSITE] Found field via Wikidata: {wikidata_fields}")
        else:
            # Fallback to semantic similarity
            similar_fields = find_semantically_similar_fields(occ_label)
            if similar_fields:
                field_results.extend(similar_fields)
                print(f"[COMPOSITE] Found similar fields for '{occ_label}': {similar_fields}")

    # Strategy 3: Try DBpedia as additional source
    if not field_results:
        dbpedia_fields = get_field_from_dbpedia(entity_label)
        if dbpedia_fields:
            field_results.extend(dbpedia_fields)
            print(f"[COMPOSITE] Found field via DBpedia: {dbpedia_fields}")

    return field_results[:5]  # Return top 5 results


def get_field_from_dbpedia(entity_label: str):
    """
    Try to get academic field from DBpedia.
    """
    entity_uri = f"http://dbpedia.org/resource/{entity_label.replace(' ', '_')}"

    query = """
    SELECT ?field ?fieldLabel WHERE {
      <%(uri)s> dbo:field ?field .
      OPTIONAL { ?field rdfs:label ?fieldLabel FILTER (lang(?fieldLabel) = 'en') }
    }
    LIMIT 5
    """ % {"uri": entity_uri}

    try:
        from entity_linking.entity_linking_utils import run_sparql_query
        results = run_sparql_query(query, endpoint="https://dbpedia.org/sparql")
        fields = []
        for res in results:
            field_uri = res.get("field", {}).get("value", "")
            field_label = res.get("fieldLabel", {}).get("value", field_uri.split("/")[-1])
            fields.append((field_uri, field_label))
        return fields
    except Exception as e:
        print(f"[DBPEDIA FIELD LOOKUP] Failed for {entity_label}: {e}")
        return []