# -----------------------------
# Helper: Decompose composite entity phrases and resolve real QIDs/PIDs
# -----------------------------
import urllib.parse
import re
import requests
from difflib import SequenceMatcher
from entity_linking.study_fields import get_academic_field_for_person

WIKIDATA_API = "https://www.wikidata.org/w/api.php"

def is_composite_label(label: str) -> bool:
    """
    Determine if a label is composite — either containing 'of' or a verb (from VERB_TO_PROPERTY),
    or being unusually long. Works for phrases like 'state flower of Arizona' and
    'michael j fox marry' alike.
    """
    if not label or len(label.split()) <= 1:
        return False

    lower = label.lower().strip()

    # --- Common pattern: "<prop> of <entity>"
    if " of " in lower or " of the " in lower:
        return True

    # --- Detect if any known verb appears
    try:
        from .verb_mappings import VERB_TO_PROPERTY  # adjust if VERB_TO_PROPERTY is local
    except ImportError:
        VERB_TO_PROPERTY = {}

    for verb in VERB_TO_PROPERTY.keys():
        if re.search(rf"\b{re.escape(verb)}\b", lower):
            return True

    # --- NEW: Detect question patterns like "X study", "X research", etc.
    study_patterns = [
        r"\bstudy\b", r"\bresearch\b", r"\bfield\b", r"\bspecializ",
        r"\bmajor\b", r"\bdiscipline\b", r"\bsubject\b"
    ]
    for pattern in study_patterns:
        if re.search(pattern, lower):
            return True

    # --- Otherwise trigger for long multiword labels (heuristic)
    return lower.count(" ") >= 2  # Reduced from 3 to catch more cases


def decompose_phrase(label: str):
    """
    Try to split composite phrases into (property_label, entity_label).

    Handles:
      • '<property> of <entity>'        → ('flower', 'Arizona')
      • '<entity> <verb>'               → ('marry', 'michael j fox')
      • '<entity> was <verb> to <target>' → ('marry', 'michael j fox')
      • '<entity> study/research'       → ('field of study', 'stephen hawking')  # NEW
    Returns (prop_label, entity_label) or (None, None).
    """
    if not label:
        return None, None

    lower = label.lower().strip()

    # --- NEW: Handle "X study/research/field" patterns first
    study_verbs = ["study", "research", "field", "specializ", "major", "discipline", "subject"]
    for verb in study_verbs:
        # Pattern: "stephen hawking study" -> ('field of study', 'stephen hawking')
        pattern1 = rf"^(?P<ent>.+?)\s+{re.escape(verb)}$"
        m1 = re.search(pattern1, lower)
        if m1:
            entity_part = m1.group("ent").strip()
            if entity_part and len(entity_part.split()) >= 1:
                return "field of study", entity_part

        # Pattern: "what did X study" -> ('field of study', 'X')
        pattern2 = rf"^what\s+(?:did|does)\s+(?P<ent>.+?)\s+{re.escape(verb)}"
        m2 = re.search(pattern2, lower)
        if m2:
            entity_part = m2.group("ent").strip()
            if entity_part and len(entity_part.split()) >= 1:
                return "field of study", entity_part

    # --- Classic "<prop> of <entity>"
    m = re.search(r"^(?P<prop>.+?)\s+of\s+(?P<ent>.+)$", lower)
    if m:
        return m.group("prop").strip(), m.group("ent").strip()

    # --- Load verbs dynamically if available
    try:
        from .verb_mappings import VERB_TO_PROPERTY
    except ImportError:
        VERB_TO_PROPERTY = {}

    # --- Verb-based "<entity> <verb>"
    tokens = lower.split()
    for verb in VERB_TO_PROPERTY.keys():
        if verb in tokens:
            idx = tokens.index(verb)
            entity = " ".join(tokens[:idx]).strip()
            if entity:
                return verb, entity

    # --- Fallback "<prop> for <entity>"
    m = re.search(r"^(?:the\s+)?(?P<prop>.+?)\s+(?:of|for)\s+(?P<ent>.+)$", lower)
    if m:
        return m.group("prop").strip(), m.group("ent").strip()

    # --- Extra: "was <verb> to" (e.g. 'was married to')
    m = re.search(r"(?P<ent>[\w\s\.\-']+?)\s+was\s+(?P<verb>\w+)\s+to", lower)
    if m:
        return m.group("verb").strip(), m.group("ent").strip()

    return None, None




def wikidata_search_label(label: str, search_type="item", limit=5):
    """
    Use Wikidata search API to find candidate QIDs or property IDs for a label.
    Adds a proper User-Agent header to avoid 403 Forbidden errors.
    search_type: "item" (for Q items) or "property" (for P properties).
    Returns list of dicts {'id': 'Q123', 'label': 'Arizona', 'description': '...'}
    """
    if not label:
        return []

    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "format": "json",
        "limit": limit,
        "type": "property" if search_type == "property" else "item"
    }

    headers = {
        "User-Agent": "EntityLinkingPipeline/1.0 (contact: your_email@example.com)"
    }

    try:
        r = requests.get(WIKIDATA_API, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("search", [])
    except requests.exceptions.RequestException as e:
        print(f"[Wikidata search] failed for '{label}': {e}")
        return []


def resolve_composite_entity(qid_label_map):
    """
    Detect and resolve composite entity phrases like 'state flower of Arizona'.
    Adds normalization for symbolic modifiers like 'state', 'national', etc.
    Now includes special handling for academic field queries.
    """
    new_map = dict(qid_label_map)
    composite_mappings = {}

    for qid, label in list(qid_label_map.items()):
        # Skip if already processed or too short
        if len(label.split()) <= 1:
            continue

        prop_label, entity_label = decompose_phrase(label)
        if not prop_label or not entity_label:
            if not is_composite_label(label):
                continue
            # Try one more time with the full label
            prop_label, entity_label = decompose_phrase(label)
            if not prop_label or not entity_label:
                continue

        print(f"[COMPOSITE] Decomposed '{label}' -> prop:'{prop_label}' entity:'{entity_label}'")

        # --- Special handling for academic field queries ---
        if prop_label.lower() in ["field of study", "study", "research", "academic field"]:
            print(f"[COMPOSITE] Academic field query detected for '{entity_label}'")

            # Resolve the entity (e.g., 'stephen hawking' -> Q17714)
            entity_candidates = wikidata_search_label(entity_label, search_type="item", limit=5)
            resolved_qid = None
            if entity_candidates:
                resolved_qid = entity_candidates[0].get("id")
                print(f"[COMPOSITE] Resolved entity '{entity_label}' -> {resolved_qid}")
            else:
                print(f"[COMPOSITE] Could not resolve entity '{entity_label}'")
                continue

            # Look for field of study properties
            field_properties = ["P101", "P106", "P69"]  # field of work, occupation, educated at
            prop_values = []

            for pid in field_properties:
                values = get_property_value(resolved_qid, pid)
                if values:
                    print(f"[COMPOSITE] Found {len(values)} values for property {pid}")
                    prop_values.extend(values)

            # If no specific field found, use comprehensive field detection
            if not prop_values:
                print(f"[COMPOSITE] No direct field properties found, using comprehensive detection...")
                prop_values = get_academic_field_for_person(resolved_qid, entity_label)

                if not prop_values:
                    print(f"[COMPOSITE] No field detected via automated methods")
                    # Optional: Add one final fallback here if needed

            # Update mappings
            if resolved_qid:
                if resolved_qid not in new_map:
                    new_map[resolved_qid] = entity_candidates[0].get("label", entity_label)
                new_map.pop(qid, None)

                composite_mappings[qid] = {
                    "resolved_qid": resolved_qid,
                    "prop_label": prop_label,
                    "entity_label": entity_label,
                    "modifier_context": "academic-field",
                    "property_values": prop_values
                }

                # Print results
                for val_id, val_label in prop_values:
                    print(f"[COMPOSITE RESULT ✅] {entity_label}'s field of study is {val_label}.")

            continue  # Skip the rest for academic field queries

        # --- Original logic for other composite types ---
        prop_lower = prop_label.lower().strip()
        modifier_context = None
        if prop_lower.startswith("state "):
            modifier_context = "state-symbol"
            base_prop = prop_lower.replace("state ", "").strip()
        elif prop_lower.startswith("national "):
            modifier_context = "national-symbol"
            base_prop = prop_lower.replace("national ", "").strip()
        elif prop_lower.startswith("official "):
            modifier_context = "official-symbol"
            base_prop = prop_lower.replace("official ", "").strip()
        else:
            base_prop = prop_lower

        print(f"[COMPOSITE] Context: {modifier_context or 'none'}, base prop: '{base_prop}'")

        # --- Resolve entity
        entity_candidates = wikidata_search_label(entity_label, search_type="item", limit=5)
        resolved_qid = None
        if entity_candidates:
            resolved_qid = entity_candidates[0].get("id")

        # --- Special-case enrichment for state/national symbols
        special_property_map = {
            "state-symbol": "P2971",  # official symbol (state/national)
            "national-symbol": "P2971",
            "official-symbol": "P2971"
        }

        prop_values = []
        if resolved_qid and modifier_context in special_property_map:
            pid = special_property_map[modifier_context]
            vals = get_property_value(resolved_qid, pid)
            if vals:
                prop_values.extend(vals)
                for _, val_label in vals:
                    print(f"[COMPOSITE RESULT ✅] {entity_label}'s {prop_label} is {val_label}.")
            else:
                print(f"[COMPOSITE] No direct {prop_label} value found for {entity_label}.")
        else:
            # --- Fallback: general property-value lookup
            try:
                from .verb_mappings import VERB_TO_PROPERTY
            except ImportError:
                VERB_TO_PROPERTY = {}

            matched_pid = None
            for verb, pid in VERB_TO_PROPERTY.items():
                if re.fullmatch(rf"{re.escape(verb)}", base_prop, flags=re.I):
                    matched_pid = pid
                    break

            if resolved_qid and matched_pid:
                # ✅ Use direct known property mapping
                vals = get_property_value(resolved_qid, matched_pid)
                if vals:
                    prop_values.extend(vals)
                    for _, val_label in vals:
                        print(f"[COMPOSITE RESULT ✅] {entity_label}'s {prop_label} is {val_label}.")
                else:
                    print(f"[COMPOSITE] No direct value found for '{prop_label}' on {entity_label}.")
            else:
                # --- Fallback: general property-value lookup
                prop_candidates_raw = find_related_property_via_wikidata(resolved_qid, base_prop)
                prop_candidates = [p["property_id"] for p in prop_candidates_raw]

                if prop_candidates:
                    for pid in prop_candidates[:5]:
                        vals = get_property_value(resolved_qid, pid)
                        if vals:
                            prop_values.extend(vals)

        # --- Update mappings for non-academic cases
        if resolved_qid:
            if resolved_qid not in new_map:
                new_map[resolved_qid] = entity_candidates[0].get("label", entity_label)
            new_map.pop(qid, None)

            # --- NEW: Try DBpedia fallback for state symbols ---
            dbpedia_values = get_dbpedia_symbol(entity_label, prop_label)
            if dbpedia_values:
                print(f"[COMPOSITE] Found {len(dbpedia_values)} DBpedia values for {entity_label} ({prop_label})")
                prop_values.extend(dbpedia_values)
            else:
                print(f"[COMPOSITE] No DBpedia values found for {entity_label} ({prop_label})")

            composite_mappings[qid] = {
                "resolved_qid": resolved_qid,
                "prop_label": prop_label,
                "entity_label": entity_label,
                "modifier_context": modifier_context,
                "property_values": prop_values
            }
        else:
            print(f"[COMPOSITE] Could not resolve entity for '{entity_label}'")

    return new_map, composite_mappings


def find_related_property_via_wikidata(entity_qid: str, prop_label: str):
    import requests, re

    if not entity_qid or not prop_label:
        return []

    # --- Context-aware token handling ---
    tokens = [t.strip().lower() for t in re.split(r"\W+", prop_label) if t.strip()]

    # If we know this came from a "state-symbol" modifier, expand search context
    extra_context_tokens = []
    if "state" in prop_label.lower() or "symbol" in prop_label.lower():
        extra_context_tokens.extend(["state", "symbol", "official", "emblem", "insignia"])

    # De-duplicate tokens while preserving order
    seen = set()
    all_tokens = [t for t in tokens + extra_context_tokens if not (t in seen or seen.add(t))]

    token_filters = " || ".join(
        [
            f'CONTAINS(LCASE(STR(?propertyLabel)), "{tok}") || '
            f'CONTAINS(LCASE(STR(?valueLabel)), "{tok}") || '
            f'CONTAINS(LCASE(STR(?valueDescription)), "{tok}")'
            for tok in all_tokens
        ]
    )

    query = f"""
    SELECT DISTINCT ?property ?propertyLabel ?value ?valueLabel ?valueDescription WHERE {{
      wd:{entity_qid} ?p ?value .
      ?property wikibase:directClaim ?p .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
      OPTIONAL {{ ?value schema:description ?valueDescription FILTER(LANG(?valueDescription) = "en") }}
      FILTER({token_filters})
    }}
    LIMIT 50
    """

    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "EntityLinkerBot/1.0 (https://example.org)"}

    try:
        r = requests.get(url, params={"query": query, "format": "json"}, headers=headers, timeout=20)
        r.raise_for_status()
        results = r.json().get("results", {}).get("bindings", [])

        candidates = [
            {
                "property_id": res["property"]["value"].split("/")[-1],
                "property_label": res["propertyLabel"]["value"],
                "value_id": res["value"]["value"].split("/")[-1],
                "value_label": res["valueLabel"]["value"],
            }
            for res in results
        ]

        # ✅ Rank by similarity between the composite phrase and property/value labels
        for c in candidates:
            prop_sim = SequenceMatcher(None, prop_label.lower(), c["property_label"].lower()).ratio()
            val_sim = SequenceMatcher(None, prop_label.lower(), c["value_label"].lower()).ratio()
            c["score"] = max(prop_sim, val_sim)

        # ✅ Keep only the most semantically similar candidates
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        top = [c for c in candidates if c["score"] > 0.35]  # adjustable threshold

        return top or candidates[:5]
    except Exception as e:
        print(f"[SPARQL lookup failed] for {entity_qid} ({prop_label}): {e}")
        return []


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

def get_dbpedia_symbol(entity_label: str, prop_label: str):
    """
    Query DBpedia for state symbols or official items like, bird, etc.
    Returns list of (value_uri, value_label).
    """
    import requests

    # normalize
    prop_label = prop_label.lower().strip()
    entity_label = entity_label.strip().title().replace(" ", "_")
    entity_uri = f"http://dbpedia.org/resource/{entity_label}"

    # common DBpedia properties that hold these facts
    candidate_props = ["dbp:flower", "dbo:symbol", "dbo:flora", "dbp:symbol"]

    query = f"""
    SELECT ?prop ?value ?valueLabel WHERE {{
      VALUES ?prop {{ {' '.join(candidate_props)} }}
      <{entity_uri}> ?prop ?value .
      OPTIONAL {{ ?value rdfs:label ?valueLabel FILTER (lang(?valueLabel) = 'en') }}
    }}
    LIMIT 10
    """

    url = "https://dbpedia.org/sparql"
    headers = {"User-Agent": "EntityLinkerBot/1.0 (https://example.org)"}

    try:
        r = requests.get(url, params={"query": query, "format": "json"}, headers=headers, timeout=15)
        r.raise_for_status()
        results = r.json().get("results", {}).get("bindings", [])

        values = []
        for res in results:
            val_uri = res["value"]["value"]
            val_label = res.get("valueLabel", {}).get("value", val_uri.split("/")[-1])
            values.append((val_uri, val_label))
        return values
    except Exception as e:
        print(f"[DBpedia lookup failed] for {entity_label}, {prop_label}: {e}")
        return []

def wikidata_search_label(label: str, search_type="item", limit=5):
    """
    Use Wikidata search API to find candidate QIDs or property IDs for a label.
    Adds a proper User-Agent header to avoid 403 Forbidden errors.
    search_type: "item" (for Q items) or "property" (for P properties).
    Returns list of dicts {'id': 'Q123', 'label': 'Arizona', 'description': '...'}
    """
    if not label:
        return []

    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "format": "json",
        "limit": limit,
        "type": "property" if search_type == "property" else "item"
    }

    headers = {
        "User-Agent": "EntityLinkingPipeline/1.0"
    }

    try:
        r = requests.get("https://www.wikidata.org/w/api.php", params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("search", [])
    except requests.exceptions.RequestException as e:
        print(f"[Wikidata search] failed for '{label}': {e}")
        return []