from collections import defaultdict
import re
from entity_linking.entity_linking_utils import *
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def extract_covid_knowledge_from_wikidata(question_text, entities_dict):
    """
    Extract COVID-19 specific knowledge from Wikidata based on question type
    and available entities.
    """
    import re
    from collections import defaultdict

    question_lower = question_text.lower()
    results = defaultdict(list)

    # Classify question type
    question_type = classify_covid_question(question_lower)
    print(f"[COVID KNOWLEDGE] Question type: {question_type}")

    # Get relevant entities
    covid_entities = identify_covid_entities(entities_dict, question_lower)

    if not covid_entities:
        print("[COVID KNOWLEDGE] No relevant COVID entities found")
        return results

    # Query Wikidata based on question type
    if question_type == "vaccine_candidates":
        results.update(query_vaccine_candidates(covid_entities))

    elif question_type == "mutations":
        results.update(query_mutations_info(covid_entities))

    elif question_type == "immune_response":
        results.update(query_immune_response(covid_entities))

    elif question_type == "superspreaders":
        results.update(query_superspreader_info(covid_entities))

    elif question_type == "symptoms":
        results.update(query_symptoms_info(covid_entities))

    elif question_type == "general_covid":
        results.update(query_general_covid_info(covid_entities, question_lower))

    return results


def classify_covid_question(question_lower):
    """Classify the type of COVID-19 question"""
    vaccine_keywords = ["vaccine", "candidate", "clinical trial", "immunization", "pfizer", "moderna", "astrazeneca"]
    mutation_keywords = ["mutation", "variant", "genome", "genomic", "strain", "spike protein", "omicron", "delta", "alpha", "beta"]
    immune_keywords = ["immune response", "antibody", "t cell", "re infection", "reinfection", "immunity", "immune", "antibodies", "igg", "igm", "t-cell", "t cell mediated", "prevent re infection", "immune protection"]
    superspreader_keywords = ["super spreader", "superspreader", "transmission", "spread event", "outbreak", "cluster"]
    symptom_keywords = ["symptom", "clinical presentation", "pneumonia", "fever", "cough", "breath", "fatigue", "headache"]
    treatment_keywords = ["treatment", "therapy", "drug", "medication", "remdesivir", "dexamethasone"]

    # Check for mutation-related questions first (since your example is about mutations)
    if any(kw in question_lower for kw in mutation_keywords):
        return "mutations"
    elif any(kw in question_lower for kw in vaccine_keywords):
        return "vaccine_candidates"
    elif any(kw in question_lower for kw in immune_keywords):
        return "immune_response"
    elif any(kw in question_lower for kw in symptom_keywords):
        return "symptoms"
    elif any(kw in question_lower for kw in superspreader_keywords):
        return "superspreaders"
    elif any(kw in question_lower for kw in treatment_keywords):
        return "general_covid"  # Could create a separate treatment category if needed
    else:
        return "general_covid"


def identify_covid_entities(entities_dict, question_lower):
    """Identify relevant COVID-19 entities from the extracted entities"""
    covid_entities = {}

    # COVID-related entity patterns - expanded and more flexible
    covid_patterns = [
        "sars-cov-2", "covid-19", "coronavirus", "2019-ncov",
        "severe acute respiratory syndrome coronavirus 2",
        "covid", "sars cov 2", "sars-cov2"
    ]

    # COVID-related keywords that might appear in entity labels
    covid_keywords = [
        "covid", "coronavirus", "sars", "pandemic", "epidemic",
        "variant", "mutation", "spike protein", "vaccine",
        "outbreak", "quarantine", "lockdown"
    ]

    for qid, label in entities_dict.items():
        label_lower = label.lower()

        # 1. Direct pattern matches (exact or partial)
        if any(pattern in label_lower for pattern in covid_patterns):
            covid_entities[qid] = label
            print(f"[COVID ENTITY] Direct pattern match: {label}")

        # 2. Keyword matches in entity labels
        elif any(keyword in label_lower for keyword in covid_keywords):
            covid_entities[qid] = label
            print(f"[COVID ENTITY] Keyword match: {label}")

        # 3. Check entity types for medical/biological relevance
        else:
            try:
                entity_types = fetch_entity_types(qid)
                type_str = ' '.join(entity_types).lower()

                # Medical/biological entity types that might be COVID-related
                medical_types = [
                    'virus', 'disease', 'pathogen', 'biological', 'medical',
                    'protein', 'gene', 'genome', 'mutation', 'variant',
                    'symptom', 'treatment', 'vaccine'
                ]

                if any(med_type in type_str for med_type in medical_types):
                    covid_entities[qid] = label
                    print(f"[COVID ENTITY] Medical type match: {label} (types: {entity_types})")

            except Exception as e:
                print(f"[COVID ENTITY] Error checking types for {qid}: {e}")

    # 4. If no entities found but question is clearly COVID-related, include all entities
    if not covid_entities and any(kw in question_lower for kw in covid_keywords + covid_patterns):
        print("[COVID ENTITY] No specific COVID entities found, but question is COVID-related - using all entities")
        covid_entities = entities_dict.copy()

    print(f"[COVID ENTITY] Identified {len(covid_entities)} COVID-related entities: {list(covid_entities.values())}")
    return covid_entities


def query_vaccine_candidates(covid_entities):
    """Query Wikidata for vaccine candidate information"""
    results = defaultdict(list)

    # Query for COVID-19 vaccines and candidates
    query = """
    SELECT ?vaccine ?vaccineLabel ?description ?clinicalTrial ?developer WHERE {
      ?vaccine wdt:P31 wd:Q87719492;  # instance of COVID-19 vaccine
               rdfs:label ?vaccineLabel.
      OPTIONAL { ?vaccine schema:description ?description. }
      OPTIONAL { ?vaccine wdt:P3098 ?clinicalTrial. }  # clinical trial
      OPTIONAL { ?vaccine wdt:P178 ?developer. }       # developer
      FILTER(LANG(?vaccineLabel) = "en")
      FILTER(LANG(?description) = "en")
    }
    LIMIT 20
    """

    try:
        vaccine_data = run_sparql_query(query)
        for item in vaccine_data:
            vaccine_info = {
                "type": "vaccine_candidate",
                "name": item.get("vaccineLabel", "Unknown"),
                "description": item.get("description", ""),
                "clinical_trial": item.get("clinicalTrial", ""),
                "developer": item.get("developerLabel", ""),
                "provenance": "wikidata_vaccine_query"
            }
            results["vaccines"].append(vaccine_info)

    except Exception as e:
        print(f"[VACCINE QUERY] Error: {e}")

    return results


def query_mutations_info(covid_entities):
    """Query Wikidata for SARS-CoV-2 mutation information using multiple strategies"""
    results = defaultdict(list)

    import time

    def run_query_with_retry(query, query_name, max_retries=3, base_delay=5):
        """Run SPARQL query with exponential backoff retry logic"""
        for attempt in range(max_retries):
            print(f"[{query_name}] Attempt {attempt + 1}/{max_retries}...")
            results = run_sparql_query(query, max_retries=1, base_delay=base_delay)

            if results is not None and len(results) > 0:
                return results

            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[{query_name}] Waiting {delay} seconds before retry...")
                time.sleep(delay)

        print(f"[{query_name}] All attempts failed")
        return []

    # Query 1: Find mutations by their TYPE (missense mutations, etc.)
    query_mutations_by_type = """
    SELECT DISTINCT ?mutation ?mutationLabel ?description ?mutationType ?mutationTypeLabel WHERE {
      # Find mutations with specific types like missense mutation
      ?mutation wdt:P31 ?mutationType;                # instance of mutation type
                wdt:P703 wd:Q82069695.                # found in taxon SARS-CoV-2

      ?mutation rdfs:label ?mutationLabel.
      ?mutationType rdfs:label ?mutationTypeLabel.

      OPTIONAL { ?mutation schema:description ?description. }

      FILTER(LANG(?mutationLabel) = "en")
      FILTER(LANG(?mutationTypeLabel) = "en")
      FILTER(LANG(?description) = "en")

      # Filter for common mutation types in SARS-CoV-2
      FILTER(?mutationType IN (
        wd:Q2656896,    # missense mutation
        wd:Q24721219,   # protein variant
        wd:Q30225616,   # biological variant
        wd:Q27895442,   # mutation of SARS-CoV-2
        wd:Q27895429,   # SARS-CoV-2 mutation
        wd:Q210961      # mutation (general)
      ))
    }
    LIMIT 30
    """

    # Query 2: Find mutations that are instances of specific mutation types
    query_mutation_instances = """
    SELECT DISTINCT ?mutation ?mutationLabel ?description ?mutationType ?mutationTypeLabel WHERE {
      # Find mutations that are instances of specific mutation subclasses
      ?mutation wdt:P31/wdt:P279* ?mutationType;      # instance of mutation type or subclass
                wdt:P703 wd:Q82069695.                # found in taxon SARS-CoV-2

      ?mutation rdfs:label ?mutationLabel.
      ?mutationType rdfs:label ?mutationTypeLabel.

      OPTIONAL { ?mutation schema:description ?description. }

      FILTER(LANG(?mutationLabel) = "en")
      FILTER(LANG(?mutationTypeLabel) = "en")
      FILTER(LANG(?description) = "en")

      # Look for mutation types that might contain SARS-CoV-2 mutations
      FILTER(CONTAINS(LCASE(?mutationTypeLabel), "mutation") || 
             CONTAINS(LCASE(?mutationTypeLabel), "variant"))
    }
    LIMIT 25
    """

    # Query 3: Find mutations through variant relationships
    query_mutations_from_variants = """
    SELECT DISTINCT ?mutation ?mutationLabel ?description ?variant ?variantLabel WHERE {
      ?variant wdt:P31 wd:Q104450895;           # instance of SARS-CoV-2 variant
               wdt:P11068 ?mutation.            # has spike protein mutation

      ?variant rdfs:label ?variantLabel.
      ?mutation rdfs:label ?mutationLabel.

      OPTIONAL { ?mutation schema:description ?description. }

      FILTER(LANG(?variantLabel) = "en")
      FILTER(LANG(?mutationLabel) = "en")
      FILTER(LANG(?description) = "en")
    }
    LIMIT 25
    """

    # Query 4: Find mutations through protein relationships
    query_mutations_from_proteins = """
    SELECT DISTINCT ?mutation ?mutationLabel ?description ?protein ?proteinLabel WHERE {
      ?mutation wdt:P31/wdt:P279* wd:Q210961;   # instance of mutation or subclass
                wdt:P703 wd:Q82069695;           # found in taxon SARS-CoV-2
                wdt:P681 ?protein.               # affects protein

      ?mutation rdfs:label ?mutationLabel.
      ?protein rdfs:label ?proteinLabel.

      OPTIONAL { ?mutation schema:description ?description. }

      FILTER(LANG(?mutationLabel) = "en")
      FILTER(LANG(?proteinLabel) = "en")
      FILTER(LANG(?description) = "en")
    }
    LIMIT 25
    """

    # Query 5: Find mutations that are biological variants
    query_biological_variants = """
    SELECT DISTINCT ?mutation ?mutationLabel ?description ?protein ?proteinLabel WHERE {
      ?mutation wdt:P31/wdt:P279* wd:Q210961;   # instance of mutation or subclass
                wdt:P703 wd:Q82069695;           # found in taxon SARS-CoV-2
                wdt:P3403 ?protein.              # biological variant of

      ?mutation rdfs:label ?mutationLabel.
      ?protein rdfs:label ?proteinLabel.

      OPTIONAL { ?mutation schema:description ?description. }

      FILTER(LANG(?mutationLabel) = "en")
      FILTER(LANG(?proteinLabel) = "en")
      FILTER(LANG(?description) = "en")
    }
    LIMIT 20
    """

    try:
        print("[MUTATIONS QUERY] Starting comprehensive mutation search...")

        # Query 1: Mutations by type (MOST IMPORTANT - includes missense mutations)
        print("[MUTATIONS QUERY] Finding mutations by type (missense, protein variants, etc.)...")
        time.sleep(1)
        type_mutations = run_query_with_retry(query_mutations_by_type, "TYPE_MUTATIONS_QUERY")
        for item in type_mutations:
            mutation_info = {
                "type": "typed_mutation",
                "name": item.get("mutationLabel", "Unknown"),
                "description": item.get("description", ""),
                "mutation_type": item.get("mutationTypeLabel", ""),
                "provenance": "wikidata_mutation_type"
            }
            results["mutations"].append(mutation_info)

        # Query 2: Mutation instances
        print("[MUTATIONS QUERY] Finding mutation instances...")
        time.sleep(1)
        instance_mutations = run_query_with_retry(query_mutation_instances, "INSTANCE_MUTATIONS_QUERY")
        for item in instance_mutations:
            mutation_info = {
                "type": "instance_mutation",
                "name": item.get("mutationLabel", "Unknown"),
                "description": item.get("description", ""),
                "mutation_type": item.get("mutationTypeLabel", ""),
                "provenance": "wikidata_mutation_instance"
            }
            results["mutations"].append(mutation_info)

        # Query 3: Mutations from variants
        print("[MUTATIONS QUERY] Finding mutations from variants...")
        time.sleep(1)
        variant_mutations = run_query_with_retry(query_mutations_from_variants, "VARIANTS_MUTATIONS_QUERY")
        for item in variant_mutations:
            mutation_info = {
                "type": "variant_associated_mutation",
                "name": item.get("mutationLabel", "Unknown"),
                "description": item.get("description", ""),
                "variant": item.get("variantLabel", ""),
                "provenance": "wikidata_variant_association"
            }
            results["mutations"].append(mutation_info)

        # Query 4: Mutations from protein relationships
        print("[MUTATIONS QUERY] Finding mutations from protein relationships...")
        time.sleep(1)
        protein_mutations = run_query_with_retry(query_mutations_from_proteins, "PROTEIN_MUTATIONS_QUERY")
        for item in protein_mutations:
            mutation_info = {
                "type": "protein_affecting_mutation",
                "name": item.get("mutationLabel", "Unknown"),
                "description": item.get("description", ""),
                "protein": item.get("proteinLabel", ""),
                "provenance": "wikidata_protein_relationship"
            }
            results["mutations"].append(mutation_info)

        # Query 5: Biological variants
        print("[MUTATIONS QUERY] Finding biological variants...")
        time.sleep(1)
        biological_mutations = run_query_with_retry(query_biological_variants, "BIOLOGICAL_VARIANTS_QUERY")
        for item in biological_mutations:
            mutation_info = {
                "type": "biological_variant",
                "name": item.get("mutationLabel", "Unknown"),
                "description": item.get("description", ""),
                "protein": item.get("proteinLabel", ""),
                "provenance": "wikidata_biological_variant"
            }
            results["mutations"].append(mutation_info)

        print(f"[MUTATIONS QUERY] Found: {len(type_mutations)} typed mutations, "
              f"{len(instance_mutations)} instance mutations, "
              f"{len(variant_mutations)} variant-associated mutations, "
              f"{len(protein_mutations)} protein-affecting mutations, "
              f"{len(biological_mutations)} biological variants")

        # Remove duplicates based on mutation QID and name
        unique_mutations = {}
        for mutation in results["mutations"]:
            # Use a combination of name and type for uniqueness
            key = f"{mutation.get('name', '')}_{mutation.get('mutation_type', '')}"
            if key and key not in unique_mutations:
                unique_mutations[key] = mutation

        results["mutations"] = list(unique_mutations.values())
        print(f"[MUTATIONS QUERY] After deduplication: {len(results['mutations'])} unique mutations")

        # Identify spike protein mutations
        spike_mutations = []
        for mutation in results["mutations"]:
            name = mutation.get("name", "").lower()
            description = mutation.get("description", "").lower()
            protein = mutation.get("protein", "").lower()
            mutation_type = mutation.get("mutation_type", "").lower()

            # Spike protein detection patterns
            spike_indicators = [
                "spike", "s protein", "s-glycoprotein", "s glycoprotein",
                "glycoprotein", "sars-cov-2 spike", "spike mutation"
            ]

            is_spike_related = any(indicator in name or indicator in description or indicator in protein
                                   for indicator in spike_indicators)

            # Also consider mutations with specific patterns
            has_mutation_pattern = any(pattern in name for pattern in ["mutation", "substitution", "variant"])

            if is_spike_related or (has_mutation_pattern and len(name) < 50):
                spike_info = mutation.copy()
                spike_info["type"] = "spike_mutation"
                spike_info["provenance"] = "pattern_identification"
                spike_mutations.append(spike_info)

        results["spike_mutations"] = spike_mutations
        print(f"[MUTATIONS QUERY] Identified {len(spike_mutations)} potential spike mutations")

        # If no mutations found, provide fallback information
        if not results["mutations"]:
            print("[MUTATIONS QUERY] No mutations found, providing fallback information...")
            fallback_info = {
                "type": "fallback_mutation_info",
                "name": "SARS-CoV-2 Genomic Mutations",
                "description": "SARS-CoV-2 accumulates various types of mutations including missense mutations in the spike protein and other genomic regions, affecting viral properties, transmission, and immune evasion",
                "common_types": "missense mutations, protein variants, biological variants",
                "provenance": "biological_knowledge_fallback"
            }
            results["fallback_info"].append(fallback_info)

    except Exception as e:
        print(f"[MUTATIONS QUERY] Overall error: {e}")
        # Add error fallback
        error_info = {
            "type": "error_fallback",
            "name": "Mutation Information",
            "description": "SARS-CoV-2 mutations occur throughout the genome with varying frequency, particularly in the spike protein region",
            "provenance": "error_recovery"
        }
        results["fallback_info"].append(error_info)

    return results


def query_immune_response(covid_entities):
    """Query Wikidata for immune response and antibody information related to COVID-19"""
    results = defaultdict(list)

    def run_simple_query(query, query_name, skip_on_error=False):
        """Run SPARQL query with minimal retry to avoid rate limiting"""
        try:
            print(f"[{query_name}] Attempting query...")
            return run_sparql_query(query)
        except Exception as e:
            print(f"[{query_name}] Query failed: {e}")
            if skip_on_error:
                print(f"[{query_name}] Skipping this query due to errors")
                return []
            return []

    # Query 1: Direct immune entities using CORRECT QIDs from your data
    query_immune_entities = """
    SELECT DISTINCT ?item ?itemLabel ?description ?type ?typeLabel WHERE {
      VALUES ?immuneType {
        wd:Q79460      # antibody (CORRECT QID from your data)
        wd:Q188930     # B-cell (CORRECT QID from your data) 
        wd:Q7003054    # neutralizing antibody (CORRECT QID from your data)
        wd:Q182581     # immunity (CORRECT QID from your data)
        wd:Q2141450    # seroconversion (CORRECT QID from your data)
        wd:Q2304808    # memory T cell (CORRECT QID from your data)
      }

      ?item wdt:P31 ?immuneType.
      ?item rdfs:label ?itemLabel.
      ?immuneType rdfs:label ?typeLabel.

      OPTIONAL { ?item schema:description ?description. }

      FILTER(LANG(?itemLabel) = "en")
      FILTER(LANG(?description) = "en")
      FILTER(LANG(?typeLabel) = "en")

      # Filter for COVID-19 related items
      FILTER(CONTAINS(LCASE(?itemLabel), "covid") || 
             CONTAINS(LCASE(?description), "covid") ||
             CONTAINS(LCASE(?itemLabel), "sars") ||
             CONTAINS(LCASE(?description), "sars") ||
             CONTAINS(LCASE(?itemLabel), "coronavirus"))
    }
    LIMIT 15
    """

    # Query 2: COVID-19 specific immune response studies and articles
    query_covid_immune_studies = """
    SELECT DISTINCT ?study ?studyLabel ?description ?pubmed WHERE {
      ?study wdt:P31 wd:Q13442814;    # scholarly article
             wdt:P921 wd:Q84263196.    # about COVID-19

      ?study rdfs:label ?studyLabel.
      OPTIONAL { ?study schema:description ?description. }
      OPTIONAL { ?study wdt:P698 ?pubmed. }  # PubMed ID

      FILTER(LANG(?studyLabel) = "en")
      FILTER(LANG(?description) = "en")

      # Filter for immune-related studies
      FILTER(CONTAINS(LCASE(?studyLabel), "immune") || 
             CONTAINS(LCASE(?description), "immune") ||
             CONTAINS(LCASE(?studyLabel), "antibod") ||
             CONTAINS(LCASE(?description), "antibod") ||
             CONTAINS(LCASE(?studyLabel), "t cell") ||
             CONTAINS(LCASE(?description), "t cell") ||
             CONTAINS(LCASE(?studyLabel), "igg") ||
             CONTAINS(LCASE(?description), "igg") ||
             CONTAINS(LCASE(?studyLabel), "igm") ||
             CONTAINS(LCASE(?description), "igm"))
    }
    LIMIT 10
    """

    # Query 3: SIMPLIFIED - Skip the complex properties query that causes 429 errors
    query_simple_immune_facts = """
    SELECT ?item ?itemLabel ?property ?propertyLabel WHERE {
      ?item wdt:P31 wd:Q79460;  # antibodies
            ?p ?object.
      ?property wikibase:directClaim ?p.
      ?property rdfs:label ?propertyLabel.
      FILTER(LANG(?propertyLabel) = "en")
      FILTER(LANG(?itemLabel) = "en")
      FILTER(CONTAINS(LCASE(?propertyLabel), "immune") || 
             CONTAINS(LCASE(?propertyLabel), "target") ||
             CONTAINS(LCASE(?propertyLabel), "response"))
    }
    LIMIT 8
    """

    try:
        print("[IMMUNE RESPONSE QUERY] Starting immune response search...")

        # Query 1: Immune entities with CORRECT QIDs (THIS WORKS WELL)
        print("[IMMUNE RESPONSE QUERY] Finding immune entities with correct QIDs...")
        entity_data = run_simple_query(query_immune_entities, "IMMUNE_ENTITIES_QUERY")
        for item in entity_data:
            entity_info = {
                "type": "immune_entity",
                "name": item.get("itemLabel", "Unknown"),
                "description": item.get("description", ""),
                "entity_type": item.get("typeLabel", ""),
                "provenance": "wikidata_immune_entities"
            }
            results["immune_entities"].append(entity_info)

        # Query 2: COVID-19 immune studies
        print("[IMMUNE RESPONSE QUERY] Finding COVID-19 immune studies...")
        study_data = run_simple_query(query_covid_immune_studies, "COVID_IMMUNE_STUDIES_QUERY")
        for item in study_data:
            study_info = {
                "type": "immune_study",
                "name": item.get("studyLabel", "Unknown"),
                "description": item.get("description", ""),
                "pubmed_id": item.get("pubmed", ""),
                "provenance": "wikidata_immune_studies"
            }
            results["immune_studies"].append(study_info)

        # Query 3: SIMPLE immune facts (skip if it fails)
        print("[IMMUNE RESPONSE QUERY] Finding simple immune facts...")
        property_data = run_simple_query(query_simple_immune_facts, "SIMPLE_IMMUNE_FACTS_QUERY", skip_on_error=True)
        for item in property_data:
            property_info = {
                "type": "immune_property",
                "name": item.get("itemLabel", ""),
                "property": item.get("propertyLabel", ""),
                "provenance": "wikidata_simple_immune_facts"
            }
            results["immune_properties"].append(property_info)

        print(f"[IMMUNE RESPONSE QUERY] Found: {len(entity_data)} immune entities, "
              f"{len(study_data)} immune studies, {len(property_data)} immune properties")

        # REMOVED: The context addition that was creating fallback sentences
        # Only include data that comes directly from Wikidata queries

    except Exception as e:
        print(f"[IMMUNE RESPONSE QUERY] Overall error: {e}")

    return results

def query_superspreader_info(covid_entities):
    """Query Wikidata for superspreader event information"""
    results = defaultdict(list)

    # Query for notable outbreak events
    query = """
    SELECT ?event ?eventLabel ?location ?date ?description WHERE {
      ?event wdt:P31 wd:Q3241045;  # instance of outbreak
             wdt:P828 ?cause.       # has cause
      FILTER(?cause = wd:Q82069695)  # SARS-CoV-2
      ?event rdfs:label ?eventLabel.
      OPTIONAL { ?event wdt:P276 ?location. }
      OPTIONAL { ?event wdt:P585 ?date. }
      OPTIONAL { ?event schema:description ?description. }
      FILTER(LANG(?eventLabel) = "en")
      FILTER(LANG(?description) = "en")
    }
    LIMIT 10
    """

    try:
        event_data = run_sparql_query(query)
        for item in event_data:
            event_info = {
                "type": "outbreak_event",
                "name": item.get("eventLabel", "Unknown"),
                "location": item.get("locationLabel", ""),
                "date": item.get("date", ""),
                "description": item.get("description", ""),
                "provenance": "wikidata_outbreak_query"
            }
            results["outbreak_events"].append(event_info)

    except Exception as e:
        print(f"[OUTBREAK QUERY] Error: {e}")

    return results


def query_symptoms_info(covid_entities):
    """Query Wikidata for COVID-19 symptom information using multiple approaches"""
    results = defaultdict(list)

    def run_symptom_query(query, query_name):
        """Helper function to run symptom queries with error handling"""
        try:
            print(f"[SYMPTOM QUERY] Running {query_name}...")
            return run_sparql_query(query)
        except Exception as e:
            print(f"[SYMPTOM QUERY] Error in {query_name}: {e}")
            return []

    # Query 1: Find symptoms through COVID-19 disease relationships
    query_symptoms_via_disease = """
    SELECT DISTINCT ?symptom ?symptomLabel ?description WHERE {
      ?covid wdt:P780 ?symptom.  # COVID-19 has symptom
      ?symptom rdfs:label ?symptomLabel.

      OPTIONAL { ?symptom schema:description ?description. }

      FILTER(LANG(?symptomLabel) = "en")
      FILTER(LANG(?description) = "en")

      # Filter for COVID-19
      VALUES ?covid { wd:Q84263196 }  # COVID-19
    }
    LIMIT 10
    """

    # Query 2: Find symptoms that are commonly associated with COVID-19
    query_common_symptoms = """
    SELECT DISTINCT ?symptom ?symptomLabel ?description WHERE {
      ?symptom wdt:P31 wd:Q169872;  # instance of symptom
               wdt:P828+ wd:Q84263196.  # associated with COVID-19

      ?symptom rdfs:label ?symptomLabel.
      OPTIONAL { ?symptom schema:description ?description. }

      FILTER(LANG(?symptomLabel) = "en")
      FILTER(LANG(?description) = "en")
    }
    LIMIT 10
    """

    # Query 3: Find symptoms through medical condition relationships
    query_symptoms_via_signs = """
    SELECT DISTINCT ?symptom ?symptomLabel ?description WHERE {
      ?covid wdt:P3483 ?symptom.  # COVID-19 has characteristic symptom
      ?symptom rdfs:label ?symptomLabel.

      OPTIONAL { ?symptom schema:description ?description. }

      FILTER(LANG(?symptomLabel) = "en")
      FILTER(LANG(?description) = "en")

      VALUES ?covid { wd:Q84263196 }  # COVID-19
    }
    LIMIT 10
    """

    # Query 4: Find symptoms that are frequently mentioned with COVID-19
    query_frequent_symptoms = """
    SELECT DISTINCT ?symptom ?symptomLabel ?description WHERE {
      ?symptom rdfs:label ?symptomLabel.
      ?symptom wdt:P828+ ?disease.

      OPTIONAL { ?symptom schema:description ?description. }

      FILTER(LANG(?symptomLabel) = "en")
      FILTER(LANG(?description) = "en")

      # Filter for COVID-19 related symptoms
      VALUES ?disease { wd:Q84263196 }  # COVID-19

      # Common COVID-19 symptom keywords in labels
      FILTER(CONTAINS(LCASE(?symptomLabel), "fever") || 
             CONTAINS(LCASE(?symptomLabel), "cough") ||
             CONTAINS(LCASE(?symptomLabel), "fatigue") ||
             CONTAINS(LCASE(?symptomLabel), "breath") ||
             CONTAINS(LCASE(?symptomLabel), "headache") ||
             CONTAINS(LCASE(?symptomLabel), "pneumonia") ||
             CONTAINS(LCASE(?symptomLabel), "anosmia") ||
             CONTAINS(LCASE(?symptomLabel), "ageusia"))
    }
    LIMIT 15
    """

    try:
        print("[SYMPTOM QUERY] Starting comprehensive symptom search...")

        # Run all symptom queries
        symptom_data1 = run_symptom_query(query_symptoms_via_disease, "symptoms_via_disease")
        symptom_data2 = run_symptom_query(query_common_symptoms, "common_symptoms")
        symptom_data3 = run_symptom_query(query_symptoms_via_signs, "symptoms_via_signs")
        symptom_data4 = run_symptom_query(query_frequent_symptoms, "frequent_symptoms")

        # Combine all results
        all_symptoms = symptom_data1 + symptom_data2 + symptom_data3 + symptom_data4

        print(f"[SYMPTOM QUERY] Found {len(all_symptoms)} total symptom entries")

        # Process symptoms and remove duplicates
        unique_symptoms = {}
        for item in all_symptoms:
            symptom_name = item.get("symptomLabel", "").strip()
            if not symptom_name or symptom_name == "Unknown":
                continue

            # Use symptom name as key for deduplication
            if symptom_name not in unique_symptoms:
                symptom_info = {
                    "type": "symptom",
                    "name": symptom_name,
                    "description": item.get("description", ""),
                    "provenance": "wikidata_symptom_query"
                }
                unique_symptoms[symptom_name] = symptom_info

        # Convert back to list and take top 5
        symptoms_list = list(unique_symptoms.values())[:5]
        results["symptoms"] = symptoms_list

        print(f"[SYMPTOM QUERY] Returning {len(symptoms_list)} unique symptoms: {[s['name'] for s in symptoms_list]}")

    except Exception as e:
        print(f"[SYMPTOM QUERY] Overall error: {e}")

    return results


def query_general_covid_info(covid_entities, question_lower):
    """Query general COVID-19 information based on question focus"""
    results = defaultdict(list)

    # Generic COVID-19 information query
    query = """
    SELECT ?covid ?covidLabel ?description ?incubation ?mortality WHERE {
      BIND(wd:Q84263196 AS ?covid)  # COVID-19
      ?covid rdfs:label ?covidLabel.
      OPTIONAL { ?covid schema:description ?description. }
      OPTIONAL { ?covid wdt:P3514 ?incubation. }  # incubation period
      OPTIONAL { ?covid wdt:P1120 ?mortality. }   # deaths
      FILTER(LANG(?covidLabel) = "en")
      FILTER(LANG(?description) = "en")
    }
    """

    try:
        covid_data = run_sparql_query(query)
        for item in covid_data:
            general_info = {
                "type": "general_info",
                "name": item.get("covidLabel", "COVID-19"),
                "description": item.get("description", ""),
                "incubation_period": item.get("incubation", ""),
                "mortality": item.get("mortality", ""),
                "provenance": "wikidata_general_query"
            }
            results["general_info"].append(general_info)

    except Exception as e:
        print(f"[GENERAL COVID QUERY] Error: {e}")

    return results


def convert_covid_knowledge_to_sentences(covid_knowledge):
    """Convert structured COVID knowledge to natural language sentences using ONLY Wikidata data"""
    sentences = []

    for category, items in covid_knowledge.items():
        print(f"[COVID SENTENCES] Processing category: {category} with {len(items)} items")

        for item in items:
            try:
                # Process immune_entities (the actual data from your Wikidata queries)
                if category == "immune_entities":
                    name = item.get('name', '').strip()
                    description = item.get('description', '').strip()
                    entity_type = item.get('entity_type', '').strip()

                    if name and name != "Unknown":
                        if description and description != "Unknown":
                            # Clean up the description
                            clean_desc = description.replace('experimental antibody treatment for COVID-19', '').strip()
                            if clean_desc:
                                sentences.append(f"{name} is a {entity_type} that {clean_desc}")
                            else:
                                sentences.append(f"{name} is a {entity_type} used in COVID-19 treatment")
                        else:
                            sentences.append(f"{name} is a {entity_type} related to COVID-19 immune response")

                # Process immune_studies
                elif category == "immune_studies":
                    name = item.get('name', '').strip()
                    description = item.get('description', '').strip()
                    pubmed_id = item.get('pubmed_id', '').strip()

                    if name and name != "Unknown":
                        pub_info = f" (PubMed ID: {pubmed_id})" if pubmed_id else ""
                        if description:
                            sentences.append(f"Study {name}{pub_info} shows that {description}")
                        else:
                            sentences.append(f"Research study {name}{pub_info} investigates COVID-19 immune response")

                # Process immune_properties
                elif category == "immune_properties":
                    subject = item.get('subject', '').strip() or item.get('name', '').strip()
                    property_name = item.get('property', '').strip()
                    obj = item.get('object', '').strip()

                    if subject and property_name:
                        if obj:
                            sentences.append(f"{subject} {property_name} {obj}")
                        else:
                            sentences.append(f"{subject} has property {property_name}")

                elif category == "symptoms":
                    name = item.get('name', '').strip()
                    description = item.get('description', '').strip()

                    if name and name != "Unknown":
                        if description:
                            sentences.append(f"{name}: {description}")
                        else:
                            sentences.append(f"Symptom: {name}")

            except Exception as e:
                print(f"[COVID SENTENCES] Error processing item in {category}: {e}")
                continue

    print(f"[COVID SENTENCES] Generated {len(sentences)} sentences from Wikidata data")
    for i, sentence in enumerate(sentences):
        print(f"[COVID SENTENCES] Sentence {i + 1}: {sentence}")

    return sentences