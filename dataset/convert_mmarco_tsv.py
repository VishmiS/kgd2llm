import csv
import os
import json
from tqdm import tqdm


def save_tsv(data, path, fieldnames):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Saved: {path}")


def create_qrels_from_json(passages_data, output_dir):
    """
    Generate qrels directly from passages (using is_selected field).
    Format: query_id, 0, passage_id, rel
    """
    os.makedirs(output_dir, exist_ok=True)
    qrels_path = os.path.join(output_dir, 'qrels.tsv')

    with open(qrels_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['query_id', 'zero', 'passage_id', 'rel'], delimiter='\t'
        )
        writer.writeheader()

        for qid, plist in tqdm(passages_data.items(), desc="Building qrels"):
            for idx, p in enumerate(plist):
                if p.get("is_selected", 0) == 1:  # positive passage
                    writer.writerow({
                        'query_id': qid,
                        'zero': 0,
                        'passage_id': f"{qid}_{idx}",  # unique passage id
                        'rel': 1
                    })

    print(f"Saved qrels file at: {qrels_path}")


def process_ms_marco_json(json_path, output_dir):
    print(f"\nProcessing: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    passages_data = data.get('passages', {})
    queries_data = data.get('query', {})

    # Corpus TSV: ['id', 'text']
    corpus = []
    for qid, plist in tqdm(passages_data.items(), desc="Building corpus"):
        for idx, p in enumerate(plist):
            corpus.append({
                'id': f"{qid}_{idx}",  # unique passage id
                'text': p.get('passage_text', '')
            })

    # Queries TSV: ['id', 'query']
    queries = [
        {'id': qid, 'query': qtext}
        for qid, qtext in tqdm(queries_data.items(), desc="Building queries")
    ]

    # Positives TSV: ['sentence1', 'sentence2', 'query_id', 'passage_id']
    positives = []
    for qid, plist in tqdm(passages_data.items(), desc="Building positives"):
        qtext = queries_data.get(qid, "")
        for idx, p in enumerate(plist):
            if p.get("is_selected", 0) == 1:  # relevant passage
                positives.append({
                    'sentence1': qtext,
                    'sentence2': p.get('passage_text', ''),
                    'query_id': qid,
                    'passage_id': f"{qid}_{idx}"
                })

    # Save to TSV
    os.makedirs(output_dir, exist_ok=True)
    save_tsv(corpus, os.path.join(output_dir, 'corpus.tsv'), ['id', 'text'])
    save_tsv(queries, os.path.join(output_dir, 'queries.tsv'), ['id', 'query'])
    save_tsv(
        positives,
        os.path.join(output_dir, 'positives.tsv'),
        ['sentence1', 'sentence2', 'query_id', 'passage_id']
    )

    # Create qrels file from JSON directly
    create_qrels_from_json(passages_data, output_dir)


# Example usage:
process_ms_marco_json('ms_marco/dev_v2.1.json', 'ms_marco/val')
process_ms_marco_json('ms_marco/train_v2.1.json', 'ms_marco/train')
process_ms_marco_json('ms_marco/eval_v2.1_public.json', 'ms_marco/test')