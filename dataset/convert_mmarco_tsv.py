import csv
import os
import json

def save_tsv(data, path, fieldnames):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Saved: {path}")

def create_qrels(data, output_dir):
    passages_data = data.get('passages', {})
    answers_data = data.get('answers', {})

    # Build a reverse map: passage_text -> list of passage_id(s)
    text_to_pid = {}
    for pid, plist in passages_data.items():
        for p in plist:
            text = p.get('passage_text', '').strip()
            if text:
                if text not in text_to_pid:
                    text_to_pid[text] = []
                text_to_pid[text].append(pid)

    qrels = []
    for qid, answers in answers_data.items():
        for ans in answers:
            ans = ans.strip()
            if ans in text_to_pid:
                for pid in text_to_pid[ans]:
                    qrels.append({
                        'query_id': qid,
                        'zero': 0,
                        'passage_id': pid,
                        'rel': 1
                    })

    # Save qrels file
    os.makedirs(output_dir, exist_ok=True)
    qrels_path = os.path.join(output_dir, 'qrels.tsv')
    with open(qrels_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['query_id', 'zero', 'passage_id', 'rel'], delimiter='\t')
        writer.writeheader()
        for row in qrels:
            writer.writerow(row)
    print(f"Saved qrels file at: {qrels_path}")

def process_ms_marco_json(json_path, output_dir):
    print(f"\nProcessing: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    passages_data = data.get('passages', {})
    queries_data = data.get('query', {})
    answers_data = data.get('answers', {})

    # Corpus TSV: ['id', 'text']
    corpus = []
    for pid, plist in passages_data.items():
        for p in plist:
            corpus.append({
                'id': pid,
                'text': p.get('passage_text', '')
            })

    # Queries TSV: ['id', 'query']
    queries = [{'id': qid, 'query': qtext} for qid, qtext in queries_data.items()]

    # Positives TSV: ['sentence1', 'sentence2']
    positives = []
    for qid, qtext in queries_data.items():
        for answer in answers_data.get(qid, []):
            positives.append({
                'sentence1': qtext,
                'sentence2': answer
            })

    # Save to TSV
    os.makedirs(output_dir, exist_ok=True)
    save_tsv(corpus, os.path.join(output_dir, 'corpus.tsv'), ['id', 'text'])
    save_tsv(queries, os.path.join(output_dir, 'queries.tsv'), ['id', 'query'])
    save_tsv(positives, os.path.join(output_dir, 'positives.tsv'), ['sentence1', 'sentence2'])

    # Create qrels file for both dev and train sets
    create_qrels(data, output_dir)

# Example usage:
process_ms_marco_json('ms_marco/dev_v2.1.json', 'ms_marco/dev')
process_ms_marco_json('ms_marco/train_v2.1.json', 'ms_marco/train')
process_ms_marco_json('ms_marco/eval_v2.1_public.json', 'ms_marco/test')
