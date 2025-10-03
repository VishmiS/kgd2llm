import csv
import os
import json
import ast
import requests


def get_label_from_wikidata(uri):
    if "wikidata.org/entity/" not in uri:
        return uri  # fallback, no change

    qid = uri.split("/")[-1]
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"

    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        label = data['entities'][qid]['labels']['en']['value']
        return label
    except Exception:
        return uri  # fallback to the URI itself if lookup fails


def get_label_from_dbpedia(uri):
    if "dbpedia.org/resource/" not in uri:
        return uri  # fallback, no change

    sparql_query = f"""
    SELECT ?label WHERE {{
        <{uri}> rdfs:label ?label .
        FILTER (lang(?label) = "en")
    }}
    """
    endpoint = "http://dbpedia.org/sparql"
    try:
        response = requests.get(endpoint, params={'query': sparql_query, 'format': 'json'}, timeout=10)
        data = response.json()
        results = data.get('results', {}).get('bindings', [])
        if results:
            return results[0]['label']['value']
    except Exception:
        pass
    return uri  # fallback


def get_label(uri):
    if "wikidata.org" in uri:
        return get_label_from_wikidata(uri)
    elif "dbpedia.org" in uri:
        return get_label_from_dbpedia(uri)
    else:
        return uri


def save_tsv(data, path, fieldnames):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Saved: {path}")


def process_custom_jsonl(input_jsonl_path, output_dir):
    print(f"\nProcessing: {input_jsonl_path}")
    os.makedirs(output_dir, exist_ok=True)

    queries = []
    corpus = []
    positives = []
    qrels = []

    passage_id_counter = 0
    query_id_counter = 0
    answer_to_pid = {}

    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            question = entry.get('question', '').strip()
            result_query_str = entry.get('result.query', '[]')

            try:
                answers = ast.literal_eval(result_query_str)
                if not isinstance(answers, list):
                    answers = []
            except Exception:
                answers = []

            filtered_answers = []
            for a in answers:
                if a is None:
                    continue
                if isinstance(a, float) and (a != a):
                    continue
                if isinstance(a, str):
                    filtered_answers.append(a.strip())
                else:
                    filtered_answers.append(str(a).strip())

            qid = str(query_id_counter)
            queries.append({'id': qid, 'query': question})
            query_id_counter += 1

            for answer in filtered_answers:
                label = get_label(answer)
                if answer not in answer_to_pid:
                    pid = str(passage_id_counter)
                    corpus.append({'id': pid, 'text': label})
                    answer_to_pid[answer] = pid
                    passage_id_counter += 1
                pid = answer_to_pid[answer]
                positives.append({'sentence1': question, 'sentence2': label})
                qrels.append({'query_id': qid, 'zero': 0, 'passage_id': pid, 'rel': 1})

    save_tsv(queries, os.path.join(output_dir, 'queries.tsv'), ['id', 'query'])
    save_tsv(corpus, os.path.join(output_dir, 'corpus.tsv'), ['id', 'text'])
    save_tsv(positives, os.path.join(output_dir, 'positives.tsv'), ['sentence1', 'sentence2'])
    save_tsv(qrels, os.path.join(output_dir, 'qrels.tsv'), ['query_id', 'zero', 'passage_id', 'rel'])


# Example usage
process_custom_jsonl('qald9plus_en_train.json', 'train')
process_custom_jsonl('qald9plus_en_validation.json', 'val')
