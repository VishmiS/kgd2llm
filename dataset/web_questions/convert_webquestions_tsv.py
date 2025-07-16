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

def process_jsonl(input_jsonl_path, output_dir):
    print(f"\nProcessing: {input_jsonl_path}")
    os.makedirs(output_dir, exist_ok=True)

    # Count lines for progress
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    queries = []
    corpus = []
    positives = []
    qrels = []

    entries = []
    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Reading input"):
            entry = json.loads(line)
            entries.append(entry)

    passage_id_counter = 0
    query_id_counter = 0
    answer_to_pid = {}

    for entry in tqdm(entries, desc="Processing entries"):
        question = entry.get('question', '').strip()
        answers = [ans.strip() for ans in entry.get('answers', [])]

        qid = str(query_id_counter)
        queries.append({'id': qid, 'query': question})
        query_id_counter += 1

        for answer in answers:
            combined_text = f"{question} {answer}"

            if combined_text not in answer_to_pid:
                pid = str(passage_id_counter)
                corpus.append({'id': pid, 'text': combined_text})
                answer_to_pid[combined_text] = pid
                passage_id_counter += 1
            else:
                pid = answer_to_pid[combined_text]

            positives.append({'sentence1': question, 'sentence2': combined_text})
            qrels.append({'query_id': qid, 'zero': 0, 'passage_id': pid, 'rel': 1})

    # Save files
    save_tsv(queries, os.path.join(output_dir, 'queries.tsv'), ['id', 'query'])
    save_tsv(corpus, os.path.join(output_dir, 'corpus.tsv'), ['id', 'text'])
    save_tsv(positives, os.path.join(output_dir, 'positives.tsv'), ['sentence1', 'sentence2'])
    save_tsv(qrels, os.path.join(output_dir, 'qrels.tsv'), ['query_id', 'zero', 'passage_id', 'rel'])

# Example usage:
# process_jsonl('web_questions_train.json', 'train')
# process_jsonl('web_questions_validation.json', 'val')
process_jsonl('web_questions_test.json', 'test')