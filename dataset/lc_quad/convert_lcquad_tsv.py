import csv
import os
import json
import time
from SPARQLWrapper import SPARQLWrapper, JSON

DEBUG_MODE = True  # Set to False to silence query output

def save_tsv(data, path, fieldnames):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Saved: {path}")

def is_valid_select_query(query):
    """Return True if this is a valid SELECT query (not ASK/CONSTRUCT etc)."""
    stripped = query.strip().lower()
    return stripped.startswith("select") and "where" in stripped

def query_wikidata(sparql_query):
    sparql_query = sparql_query.strip().replace('\u00a0', ' ')  # Normalize non-breaking spaces

    if not is_valid_select_query(sparql_query):
        print(f"Skipping non-SELECT query:\n{sparql_query}\n")
        return []

    if DEBUG_MODE:
        print(f"\nRunning SPARQL query:\n{sparql_query}")

    endpoint_url = "https://query.wikidata.org/sparql"
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()

        if "boolean" in results:
            print("Warning: ASK query detected; skipping.")
            print(json.dumps(results, indent=2))
            return []

        if "results" not in results:
            print("Warning: No 'results' key in response. Full response:")
            print(json.dumps(results, indent=2))
            return []

        answers = []
        for result in results["results"]["bindings"]:
            for var in result:
                answers.append(result[var]['value'])

        return answers

    except Exception as e:
        print(f"SPARQL query failed: {e}")
        return []

def process_jsonl(input_jsonl_path, output_dir):
    print(f"\nProcessing: {input_jsonl_path}")
    os.makedirs(output_dir, exist_ok=True)

    queries = []
    corpus = []
    positives = []
    qrels = []

    query_id_counter = 0
    answer_id_counter = 0
    answer_to_id = {}

    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
                continue

            question = entry.get('question', '').strip()
            uid = entry.get('uid')
            sparql_query = entry.get('sparql_wikidata', '').strip()

            if not sparql_query:
                print(f"No SPARQL query found for uid {uid}, skipping...")
                continue

            answers = query_wikidata(sparql_query)
            time.sleep(1.1)  # Respect Wikidata's rate limits

            if not answers:
                print(f"No answers found for uid {uid}, skipping...\nQuery:\n{sparql_query}\n")
                continue

            answer_text = " | ".join(answers).strip()
            if not answer_text:
                continue

            qid = str(query_id_counter)
            queries.append({'id': qid, 'query': question})
            query_id_counter += 1

            if answer_text not in answer_to_id:
                aid = str(answer_id_counter)
                corpus.append({'id': aid, 'text': answer_text})
                answer_to_id[answer_text] = aid
                answer_id_counter += 1
            aid = answer_to_id[answer_text]

            positives.append({'sentence1': question, 'sentence2': answer_text})
            qrels.append({'query_id': qid, 'zero': 0, 'passage_id': aid, 'rel': 1})

    # Save all output TSVs
    save_tsv(queries, os.path.join(output_dir, 'queries.tsv'), ['id', 'query'])
    save_tsv(corpus, os.path.join(output_dir, 'corpus.tsv'), ['id', 'text'])
    save_tsv(positives, os.path.join(output_dir, 'positives.tsv'), ['sentence1', 'sentence2'])
    save_tsv(qrels, os.path.join(output_dir, 'qrels.tsv'), ['query_id', 'zero', 'passage_id', 'rel'])

    print(f"\nSummary for {input_jsonl_path}")
    print(f" - Queries processed: {len(queries)}")
    print(f" - Unique answers (corpus): {len(corpus)}")
    print(f" - Positive pairs: {len(positives)}")
    print(f" - Qrels entries: {len(qrels)}")

# Example usage
process_jsonl('lcquad2_train.json', 'train')
process_jsonl('lcquad2_validation.json', 'val')
