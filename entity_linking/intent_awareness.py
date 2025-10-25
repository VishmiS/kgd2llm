# intent_awareness.py

from sentence_transformers import SentenceTransformer, util
import re

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def analyze_question_intent(question):
    """Analyze question to determine what type of answer is expected"""
    q_lower = question.lower()

    intent = {
        "is_location": any(kw in q_lower for kw in ["where", "location", "place", "city", "country"]),
        "is_time": any(kw in q_lower for kw in ["when", "date", "year", "time"]),
        "is_person": any(kw in q_lower for kw in ["who", "person", "actor", "director"]),
        "is_thing": any(kw in q_lower for kw in ["what", "which"]),
        "is_reason": any(kw in q_lower for kw in ["why", "reason"]),
        "is_manner": any(kw in q_lower for kw in ["how"]),
    }

    # Enhanced detection for "where + study" pattern
    if "where" in q_lower and any(edu_kw in q_lower for edu_kw in
                                  ["study", "educated", "school", "university", "college", "learn", "trained"]):
        intent["expected_answer_type"] = "educational_institution"
        intent["study_context"] = True
    elif "where" in q_lower:
        intent["expected_answer_type"] = "location"

    return intent


def get_priority_properties_for_intent(intent):
    """Return property priorities based on question intent"""
    if intent.get("expected_answer_type") == "educational_institution":
        return {
            "wikidata": ["P69", "P937", "P1416"],  # educated at, work location, training
            "dbpedia": ["almaMater", "education", "training", "school"],
            "semantic_terms": ["educated at", "studied at", "alma mater", "university", "college", "school"]
        }
    elif intent.get("expected_answer_type") == "location":
        return {
            "wikidata": ["P19", "P20", "P276", "P740"],
            # place of birth, place of death, location, location of formation
            "dbpedia": ["birthPlace", "deathPlace", "location", "hometown"],
            "semantic_terms": ["location", "place", "city", "country"]
        }
    else:
        return {
            "wikidata": [],
            "dbpedia": [],
            "semantic_terms": []
        }


def enhance_composite_resolution_for_intent(composite_mappings, intent):
    """Adjust composite resolution based on question intent"""
    enhanced_mappings = {}

    for orig_id, mapping in composite_mappings.items():
        prop_label = mapping.get("prop_label", "").lower()
        entity_label = mapping.get("entity_label", "")

        # CRITICAL FIX: For "where + study" questions, override the composite mapping
        if intent.get("expected_answer_type") == "educational_institution" and "study" in prop_label:
            enhanced_mapping = mapping.copy()
            # Override to use educational institution properties instead of field of study
            enhanced_mapping["priority_properties"] = ["P69", "almaMater", "training", "school"]
            enhanced_mapping["semantic_target"] = "educational_institution"
            enhanced_mapping["original_prop_label"] = prop_label  # Keep original for reference
            enhanced_mapping["effective_prop_label"] = "educated at"  # Map "study" to "educated at"
            enhanced_mappings[orig_id] = enhanced_mapping
            print(f"[INTENT OVERRIDE] Mapping 'study' → 'educated at' for educational institution search")
        else:
            enhanced_mappings[orig_id] = mapping

    return enhanced_mappings


def is_location_like_value(value, use_cache=True):
    """Hybrid approach using multiple signals"""
    value_str = str(value).strip()

    # Quick rejection of obvious non-locations
    if (re.search(r"Point\([^)]+\)", value_str) or
            re.match(r"^-?\d+\.\d+$", value_str) or
            re.match(r"^\d+$", value_str) or
            re.search(r"\d{4}-\d{2}-\d{2}", value_str) or
            len(value_str) < 2):
        return False

    # Signal 1: Structural patterns (no hard-coded names)
    structural_indicators = [
        # Geographic administrative divisions
        "city", "county", "state", "province", "country", "nation",
        "town", "village", "municipality", "borough",
        # Infrastructure
        "airport", "station", "port", "harbor",
        # Institutions
        "university", "college", "academy", "institute",
        "school", "campus",
        # Facilities
        "prison", "jail", "museum", "theatre", "hospital", "park",
        "stadium", "arena"
    ]

    if any(indicator in value_str.lower() for indicator in structural_indicators):
        return True

    # Signal 2: Capitalization pattern (proper nouns)
    words = value_str.split()
    if len(words) <= 5:  # Reasonable for location names
        capitalized_words = sum(1 for word in words if word and word[0].isupper())
        if capitalized_words >= len(words) * 0.6:  # Mostly capitalized
            return True

    # Signal 3: Semantic similarity to location concepts
    location_concepts = [
        "geographic location place", "city town village",
        "country nation state", "building facility structure"
    ]

    value_emb = embedding_model.encode(value_str, convert_to_tensor=True)
    concept_embs = [embedding_model.encode(concept, convert_to_tensor=True)
                    for concept in location_concepts]

    max_similarity = max(util.cos_sim(value_emb, concept_emb).item()
                         for concept_emb in concept_embs)

    return max_similarity > 0.55

def clean_answer(value):
    """Clean and format answer text"""
    if not value:
        return ""

    value_str = str(value).strip()

    # Remove quotes and extra punctuation
    value_str = re.sub(r'^["\']|["\']$', '', value_str)

    # Remove trailing periods unless it's an abbreviation
    if value_str.endswith('.') and len(value_str) > 2 and not value_str[-2].isupper():
        value_str = value_str[:-1]

    # Clean timestamps (e.g., 1977-04-21T00:00:00Z → 1977-04-21)
    value_str = re.sub(r"T\d{2}:\d{2}:\d{2}Z", "", value_str)

    return value_str.strip()

def filter_facts_with_intent(facts_dict, intent, question_embedding=None, similarity_threshold=0.5):
    """Filter facts based on question intent and semantic relevance"""
    filtered_results = {}
    priority_props = get_priority_properties_for_intent(intent)

    for entity_id, facts in facts_dict.items():
        filtered_facts = []

        for fact in facts:
            if isinstance(fact, dict):
                prop = fact.get("property", "")
                value = fact.get("value", "")

                # Priority 1: Exact property matches for educational institutions
                if intent.get("expected_answer_type") == "educational_institution":
                    if prop in priority_props["wikidata"] or any(
                            p in str(prop).lower() for p in priority_props["dbpedia"]):
                        if is_institution_like_value(value):  # Only include if it's actually an institution
                            filtered_facts.append(fact)
                            continue

                # Priority 2: Semantic similarity to expected answer type
                if question_embedding is not None:
                    value_emb = embedding_model.encode(str(value), convert_to_tensor=True)
                    similarity = util.cos_sim(question_embedding, value_emb).item()

                    # For educational institution questions, prioritize institution-like values
                    if intent.get("expected_answer_type") == "educational_institution" and is_institution_like_value(
                            value):
                        if similarity > 0.4:
                            filtered_facts.append(fact)
                            continue
                    # General semantic filtering using the threshold parameter
                    elif similarity > similarity_threshold:
                        filtered_facts.append(fact)
                        continue

            elif isinstance(fact, tuple) and len(fact) == 2:
                prop, value = fact
                # Apply similar logic for tuple format facts
                if intent.get("expected_answer_type") == "educational_institution":
                    if prop in priority_props["wikidata"] or any(
                            p in str(prop).lower() for p in priority_props["dbpedia"]):
                        if is_institution_like_value(value):
                            filtered_facts.append(fact)
                            continue

                if question_embedding is not None:
                    value_emb = embedding_model.encode(str(value), convert_to_tensor=True)
                    similarity = util.cos_sim(question_embedding, value_emb).item()

                    if intent.get("expected_answer_type") == "educational_institution" and is_institution_like_value(
                            value):
                        if similarity > 0.4:
                            filtered_facts.append(fact)
                            continue
                    # General semantic filtering using the threshold parameter
                    elif similarity > similarity_threshold:
                        filtered_facts.append(fact)
                        continue

        filtered_results[entity_id] = filtered_facts

    return filtered_results


def find_direct_answer_with_intent(question_text, wikidata_facts, dbpedia_facts, intent):
    """Extract answers considering question intent"""
    q_lower = question_text.lower()

    if intent.get("expected_answer_type") == "educational_institution":
        print(f"[DIRECT ANSWER] Looking for educational institutions...")

        # Look for educational institutions in facts
        for fact_source_name, fact_source in [("Wikidata", wikidata_facts), ("DBpedia", dbpedia_facts)]:
            for entity_id, facts in fact_source.items():
                for fact in facts:
                    if isinstance(fact, dict):
                        prop = fact.get("property", "")
                        value = fact.get("value", "")

                        # Check if this is an educational property AND the value is an institution
                        if is_educational_property(prop) and is_institution_like_value(value):
                            print(f"[DIRECT ANSWER FOUND] {fact_source_name}: {value} (property: {prop})")
                            return clean_answer(value)

                    elif isinstance(fact, tuple) and len(fact) == 2:
                        prop, value = fact
                        if is_educational_property(prop) and is_institution_like_value(value):
                            print(f"[DIRECT ANSWER FOUND] {fact_source_name}: {value} (property: {prop})")
                            return clean_answer(value)

    elif intent.get("expected_answer_type") == "location":
        # Look for locations in facts
        for fact_source in [wikidata_facts, dbpedia_facts]:
            for entity_id, facts in fact_source.items():
                for fact in facts:
                    if isinstance(fact, dict):
                        prop = fact.get("property", "")
                        value = fact.get("value", "")

                        if is_location_property(prop) and is_location_like_value(value):
                            return clean_answer(value)

                    elif isinstance(fact, tuple) and len(fact) == 2:
                        prop, value = fact
                        if is_location_property(prop) and is_location_like_value(value):
                            return clean_answer(value)

    # print("[DIRECT ANSWER] No suitable educational institution found")
    return None


def is_educational_property(prop):
    """Check if property indicates education - FIXED to prioritize institutions over fields"""
    # Primary educational institution properties
    institution_props = ["P69", "almaMater", "training", "school", "educated at"]
    # Secondary field of study properties (avoid these for "where" questions)
    field_of_study_props = ["P101", "field of study", "academic discipline"]

    prop_str = str(prop).lower()

    # Prioritize institution properties
    if any(indicator in prop_str for indicator in institution_props):
        return True
    # Avoid field of study properties for location questions
    elif any(indicator in prop_str for indicator in field_of_study_props):
        return False
    # Fallback to general education terms
    else:
        edu_indicators = ["education", "study", "college", "university", "academy"]
        return any(indicator in prop_str for indicator in edu_indicators)


def is_location_property(prop):
    """Check if property indicates location"""
    location_indicators = ["P19", "P20", "P276", "P740", "birthPlace", "deathPlace", "location", "hometown"]
    prop_str = str(prop).lower()
    return any(indicator in prop_str for indicator in location_indicators)


def is_institution_like_value(value):
    """Check if value looks like an educational institution - IMPROVED"""
    value_str = str(value).lower()

    # Positive indicators - must contain these
    institution_indicators = ["university", "college", "academy", "institute", "school", "faculty"]

    # Negative indicators - avoid these (they're fields, not institutions)
    field_indicators = ["art of", "painting", "sculpture", "jewelry", "direction", "screenwriter",
                        "photographer", "actor", "writer", "illustrator", "painter", "sculptor",
                        "graphic artist", "jewelry designer", "film director"]

    # Check if it looks like an institution AND doesn't look like a field
    has_institution_indicator = any(indicator in value_str for indicator in institution_indicators)
    has_field_indicator = any(indicator in value_str for indicator in field_indicators)

    # Also accept values that contain location names (like "Madrid, Spain")
    has_location = any(loc in value_str for loc in ["madrid", "london", "paris", "new york", "berlin"])

    return (has_institution_indicator or has_location) and not has_field_indicator and len(value_str) > 3


def detect_property_semantic_type(property_label, value):
    """Detect if a property-value pair represents a location"""
    location_indicators = [
        "place", "location", "city", "country", "born", "died", "based",
        "headquarters", "residence", "hometown"
    ]

    institution_indicators = [
        "university", "college", "academy", "institute", "school"
    ]

    prop_lower = str(property_label).lower()
    val_lower = str(value).lower()

    if any(indicator in prop_lower for indicator in location_indicators):
        return "location"
    elif any(indicator in prop_lower for indicator in institution_indicators):
        return "institution"
    elif any(indicator in val_lower for indicator in institution_indicators):
        return "institution"

    return "other"