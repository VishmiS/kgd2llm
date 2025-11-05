import csv
import os
import json
import re
import requests
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


# ============================================================
# Utility Functions
# ============================================================

def count_tokens(text: str) -> int:
    """Simple token counter (approximate using whitespace splitting)."""
    if not text:
        return 0
    return len(text.split())


def normalize_text(text: str) -> str:
    """
    Robust text normalization for scientific/medical text with strict ASCII filtering.

    Steps:
    0. Pre-clean multiline text (remove \n, \t)
    1. Lowercase text
    2. Remove URLs
    3. Remove DOI, citations, or reference patterns like [1], (Smith et al., 2020)
    4. Remove special characters and punctuation (keep alphanumeric + space)
    5. Convert to ASCII, replacing or removing non-ASCII characters
    6. Normalize whitespace
    7. Strip leading/trailing spaces
    """
    if not text or not isinstance(text, str):
        return ""

    # --- Pre-clean multiline & tabbed text ---
    # Replace newlines, tabs, carriage returns with a single space
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # Replace bullet-like list markers (e.g. "* something") with just the word
    text = re.sub(r'^\s*[\*\-\u2022]+\s*', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove DOI patterns
    text = re.sub(r'doi:\s*\S+', '', text)

    # Remove citations like [1], [12-15], (Smith et al., 2020)
    text = re.sub(r'\[\d+(?:[-,]\d+)*\]', '', text)
    text = re.sub(r'\([a-zA-Z\s]+ et al\.,?\s*\d{4}\)', '', text)

    # Convert to ASCII, replacing non-ASCII characters with space
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Remove non-alphanumeric characters (keep spaces)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Normalize multiple spaces again
    text = re.sub(r'\s+', ' ', text).strip()

    return text



def save_tsv(data, path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Helper: replace tabs/newlines with spaces in all string values
    def sanitize_row(row):
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, str):
                v = v.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                v = re.sub(r'\s+', ' ', v).strip()
            clean_row[k] = v
        return clean_row

    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t', quoting=csv.QUOTE_NONE, escapechar='\\')
        writer.writeheader()
        for row in data:
            writer.writerow(sanitize_row(row))

    print(f"✅ Saved: {path} ({len(data)} records)")



# ============================================================
# DBpedia Linking + Note Retrieval (with caching)
# ============================================================

CACHE_PATH = "dbpedia_entity_cache.json"


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def link_to_dbpedia(entity_name, cache):
    """Link a short answer string to a DBpedia entity using DBpedia Spotlight API."""
    if entity_name in cache and "dbpedia_uri" in cache[entity_name]:
        return cache[entity_name]["dbpedia_uri"]

    url = "https://api.dbpedia-spotlight.org/en/annotate"
    headers = {"Accept": "application/json"}
    params = {"text": entity_name, "confidence": 0.5}

    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            resources = data.get("Resources", [])
            if resources:
                uri = resources[0]["@URI"]
                cache[entity_name] = {"dbpedia_uri": uri}
                return uri
    except Exception:
        pass

    cache[entity_name] = {"dbpedia_uri": None}
    return None


def fetch_dbpedia_note(dbpedia_uri, cache, entity_name):
    """Fetch short note for DBpedia entity (from comment or abstract)."""
    if not dbpedia_uri:
        return None

    if entity_name in cache and "note" in cache[entity_name]:
        return cache[entity_name]["note"]

    sparql_url = "https://dbpedia.org/sparql"
    query = f"""
    SELECT ?comment ?abstract WHERE {{
      OPTIONAL {{ <{dbpedia_uri}> rdfs:comment ?comment . FILTER (lang(?comment) = 'en') }}
      OPTIONAL {{ <{dbpedia_uri}> dbo:abstract ?abstract . FILTER (lang(?abstract) = 'en') }}
    }} LIMIT 1
    """

    note = None
    try:
        r = requests.get(sparql_url, params={'query': query, 'format': 'json'}, timeout=10)
        if r.status_code == 200:
            results = r.json().get("results", {}).get("bindings", [])
            if results:
                comment = results[0].get("comment", {}).get("value")
                abstract = results[0].get("abstract", {}).get("value")
                note = comment or abstract
    except Exception:
        pass

    cache.setdefault(entity_name, {})["note"] = note
    return note


# ============================================================
# Process JSONL and Create Dataset
# ============================================================
def process_jsonl(input_jsonl_path, output_dir, cache, split_offset=0):
    print(f"\n📘 Processing: {input_jsonl_path}")

    queries, corpus, positives, qrels = [], [], [], []

    with open(input_jsonl_path, 'r', encoding='utf-8') as f:
        entries = [json.loads(line) for line in f]

    # Use large offsets to ensure unique IDs across splits
    passage_id_counter = split_offset * 100000
    query_id_counter = split_offset * 100000

    # Use a split-specific answer_to_pid to avoid conflicts
    answer_to_pid = {}

    for entry in tqdm(entries, desc="Processing entries"):
        question = normalize_text(entry.get('question', '').strip())
        # Filter out empty answers after normalization
        answers = [normalize_text(ans.strip()) for ans in entry.get('answers', [])]
        answers = [ans for ans in answers if ans]  # Remove empty strings

        if not question or not answers:
            continue

        qid = str(query_id_counter)
        queries.append({'id': qid, 'query': question})
        query_id_counter += 1

        for answer in answers:
            # This check should already be handled above, but keep for safety
            if not answer or answer.isspace():
                continue

            # 🔥 ADD: Strip answer if longer than 300 tokens
            if count_tokens(answer) > 300:
                # Truncate to 300 tokens by taking first 300 words
                tokens = answer.split()[:300]
                answer = ' '.join(tokens)
                print(f"⚠️  Truncated answer from {count_tokens(answer)} to 300 tokens")

            dbpedia_uri = link_to_dbpedia(answer, cache)
            note = fetch_dbpedia_note(dbpedia_uri, cache, answer) if dbpedia_uri else None

            # Combine answer + note for semantic richness
            combined_text = answer
            if note:
                combined_text += f" , {note}"
            combined_text = normalize_text(combined_text)

            # 🔥 CRITICAL FIX: Skip if combined_text becomes empty after normalization
            if not combined_text or combined_text.isspace():
                print(f"⚠️  Skipping empty combined_text for answer: '{answer[:50]}...'")
                continue

            # 🔥 ADD: Check combined text length too
            if count_tokens(combined_text) > 300:
                tokens = combined_text.split()[:300]
                combined_text = ' '.join(tokens)
                print(f"⚠️  Truncated combined text to 300 tokens")

            # Final safety check after truncation
            if not combined_text or combined_text.isspace():
                print(f"⚠️  Skipping empty text after truncation for answer: '{answer[:50]}...'")
                continue

            if answer not in answer_to_pid:
                pid = str(passage_id_counter)
                combined_text = re.sub(r'[\r\n\t]+', ' ', combined_text).strip()
                combined_text = re.sub(r'^\s*[\*\-\u2022]+\s*', '', combined_text)
                corpus_entry = {
                    'id': pid,
                    'text': combined_text
                }
                # Only add dbpedia_uri and note if they have meaningful values
                if dbpedia_uri:
                    corpus_entry['dbpedia_uri'] = dbpedia_uri
                if note:
                    corpus_entry['note'] = note
                corpus.append(corpus_entry)
                answer_to_pid[answer] = pid
                passage_id_counter += 1
            else:
                pid = answer_to_pid[answer]

            pos_entry = {
                'sentence1': question,
                'sentence2': combined_text,
                'query_id': qid,
                'passage_id': pid
            }
            # Only add dbpedia_uri and note if they have meaningful values
            if dbpedia_uri:
                pos_entry['dbpedia_uri'] = dbpedia_uri
            if note:
                pos_entry['note'] = note
            positives.append(pos_entry)
            qrels.append({'query_id': qid, 'passage_id': pid, 'rel': 1})

    return queries, corpus, positives, qrels

def save_individual_splits(output_dir, split_name, queries, corpus, positives, qrels):
    """Save individual split data (train/val/test)"""
    # Remove duplicate qrels within the same split
    seen_qrels = set()
    unique_qrels = []

    for qrel in qrels:
        qrel_key = (qrel['query_id'], qrel['passage_id'])
        if qrel_key not in seen_qrels:
            unique_qrels.append(qrel)
            seen_qrels.add(qrel_key)

    qrels = unique_qrels

    # Rename query IDs to match fieldnames
    queries_renamed = [{'query_id': q['id'], 'query': q['query']} for q in queries]
    # Handle corpus with optional fields
    # Handle corpus with optional fields
    # 🔒 SAFE CORPUS CLEANING
    corpus_renamed = []
    none_text_count = 0

    for c in corpus:
        text = c.get('text', '')
        if text is None or not isinstance(text, str) or text.strip() == '':
            none_text_count += 1
            print(f"🚨 WARNING: Invalid or empty text in corpus entry - corpus_id: {c.get('id')}, skipping.")
            continue

        # Pre-clean text for safe TSV export (remove tabs and newlines)
        text = re.sub(r'[\r\n\t]+', ' ', text).strip()
        text = re.sub(r'\s+', ' ', text)

        corpus_entry = {'corpus_id': c['id'], 'text': text}
        if 'dbpedia_uri' in c and c['dbpedia_uri']:
            corpus_entry['dbpedia_uri'] = c['dbpedia_uri']
        if 'note' in c and c['note']:
            corpus_entry['note'] = c['note']
        corpus_renamed.append(corpus_entry)

    if none_text_count > 0:
        print(f"🚨 FILTERED OUT {none_text_count} corpus entries with None text values!")

    split_folder = os.path.join(output_dir, split_name)
    os.makedirs(split_folder, exist_ok=True)

    # Save split data
    save_tsv(queries_renamed, os.path.join(split_folder, 'queries.tsv'), ['query_id', 'query'])
    # Determine fieldnames based on available data
    corpus_fieldnames = ['corpus_id', 'text']
    positives_fieldnames = ['sentence1', 'sentence2', 'query_id', 'passage_id']

    # Check if any entry has optional fields and add them to fieldnames
    if any('dbpedia_uri' in c for c in corpus_renamed):
        corpus_fieldnames.append('dbpedia_uri')
    if any('note' in c for c in corpus_renamed):
        corpus_fieldnames.append('note')

    if any('dbpedia_uri' in p for p in positives):
        positives_fieldnames.append('dbpedia_uri')
    if any('note' in p for p in positives):
        positives_fieldnames.append('note')

    save_tsv(corpus_renamed, os.path.join(split_folder, 'corpus.tsv'), corpus_fieldnames)
    save_tsv(positives, os.path.join(split_folder, 'positives.tsv'), positives_fieldnames)


    save_tsv(qrels, os.path.join(split_folder, 'qrels.tsv'), ['query_id', 'passage_id', 'rel'])

    # Create and save summary for this split - with deduplication
    qrels_df = pd.DataFrame(qrels).rename(columns={"passage_id": "corpus_id"})
    queries_df = pd.DataFrame(queries_renamed)[["query_id", "query"]]

    # Create corpus DataFrame with available columns
    corpus_cols = ["corpus_id", "text"]
    if any('dbpedia_uri' in c for c in corpus_renamed):
        corpus_cols.append("dbpedia_uri")
    if any('note' in c for c in corpus_renamed):
        corpus_cols.append("note")
    corpus_df = pd.DataFrame(corpus_renamed)[corpus_cols]

    summary_df = pd.merge(qrels_df, queries_df, on="query_id", how="inner")
    summary_df = pd.merge(summary_df, corpus_df, on="corpus_id", how="inner")
    summary_df = summary_df.drop_duplicates(subset=['query_id', 'corpus_id'])

    summary_df.to_csv(os.path.join(split_folder, "summary.tsv"), sep="\t", index=False)

    print(f"\n📊 {split_name.upper()} split stats:")
    print(f" - Queries: {len(queries)}")
    print(f" - Corpus passages: {len(corpus)}")
    print(f" - Positives: {len(positives)}")
    print(f" - Unique Qrels: {len(qrels)}")
    print(f" - Summary records: {len(summary_df)}")


def save_combined_full_dataset(output_dir, all_data):
    """Save combined dataset from all splits into full/ folder"""
    print(f"\n🔄 Combining all splits into full dataset...")

    # Combine all data with proper deduplication
    combined_queries = []
    combined_corpus = []
    combined_positives = []
    combined_qrels = []

    # Use sets to track unique items across all splits
    seen_query_ids = set()
    seen_corpus_ids = set()
    seen_qrel_pairs = set()  # Track (query_id, passage_id) pairs to avoid duplicates

    for split_name, (queries, corpus, positives, qrels) in all_data.items():
        # Add queries (deduplicate by query_id)
        for query in queries:
            if query['id'] not in seen_query_ids:
                combined_queries.append(query)
                seen_query_ids.add(query['id'])

        # Add corpus (deduplicate by corpus_id)
        for corp in corpus:
            if corp['id'] not in seen_corpus_ids:
                combined_corpus.append(corp)
                seen_corpus_ids.add(corp['id'])

        # Add positives (no deduplication needed as they're just pairs)
        combined_positives.extend(positives)

        # Add qrels (deduplicate by (query_id, passage_id) pairs)
        for qrel in qrels:
            qrel_key = (qrel['query_id'], qrel['passage_id'])
            if qrel_key not in seen_qrel_pairs:
                combined_qrels.append(qrel)
                seen_qrel_pairs.add(qrel_key)

    # Rename for final output
    queries_renamed = [{'query_id': q['id'], 'query': q['query']} for q in combined_queries]
    # Handle corpus with optional fields
    # Handle corpus with optional fields
    # 🔒 SAFE CORPUS CLEANING
    corpus_renamed = []
    none_text_count = 0

    for c in combined_corpus:
        text = c.get('text', '')
        if text is None or not isinstance(text, str) or text.strip() == '':
            none_text_count += 1
            print(f"🚨 WARNING: Invalid or empty text in combined corpus entry - corpus_id: {c.get('id')}, skipping.")
            continue

        # Pre-clean text for safe TSV export (remove tabs and newlines)
        text = re.sub(r'[\r\n\t]+', ' ', text).strip()
        text = re.sub(r'\s+', ' ', text)

        corpus_entry = {'corpus_id': c['id'], 'text': text}
        if 'dbpedia_uri' in c and c['dbpedia_uri']:
            corpus_entry['dbpedia_uri'] = c['dbpedia_uri']
        if 'note' in c and c['note']:
            corpus_entry['note'] = c['note']
        corpus_renamed.append(corpus_entry)

    if none_text_count > 0:
        print(f"🚨 FILTERED OUT {none_text_count} combined corpus entries with None text values!")

    # Save combined full dataset
    full_folder = os.path.join(output_dir, 'full')
    save_tsv(queries_renamed, os.path.join(full_folder, 'queries.tsv'), ['query_id', 'query'])
    # Determine fieldnames based on available data in combined dataset
    corpus_fieldnames = ['corpus_id', 'text']
    positives_fieldnames = ['sentence1', 'sentence2', 'query_id', 'passage_id']

    # Check if any entry has optional fields and add them to fieldnames
    if any('dbpedia_uri' in c for c in corpus_renamed):
        corpus_fieldnames.append('dbpedia_uri')
    if any('note' in c for c in corpus_renamed):
        corpus_fieldnames.append('note')

    if any('dbpedia_uri' in p for p in combined_positives):
        positives_fieldnames.append('dbpedia_uri')
    if any('note' in p for p in combined_positives):
        positives_fieldnames.append('note')

    save_tsv(corpus_renamed, os.path.join(full_folder, 'corpus.tsv'), corpus_fieldnames)
    save_tsv(combined_positives, os.path.join(full_folder, 'positives.tsv'), positives_fieldnames)
    save_tsv(combined_qrels, os.path.join(full_folder, 'qrels.tsv'), ['query_id', 'passage_id', 'rel'])

    # Create and save combined summary - with proper deduplication
    qrels_df = pd.DataFrame(combined_qrels).rename(columns={"passage_id": "corpus_id"})
    queries_df = pd.DataFrame(queries_renamed)[["query_id", "query"]]

    # Create corpus DataFrame with available columns
    corpus_cols = ["corpus_id", "text"]
    if any('dbpedia_uri' in c for c in corpus_renamed):
        corpus_cols.append("dbpedia_uri")
    if any('note' in c for c in corpus_renamed):
        corpus_cols.append("note")
    corpus_df = pd.DataFrame(corpus_renamed)[corpus_cols]
    # Merge with inner joins to ensure only valid pairs
    summary_df = pd.merge(qrels_df, queries_df, on="query_id", how="inner")
    summary_df = pd.merge(summary_df, corpus_df, on="corpus_id", how="inner")

    # Remove any potential duplicates in the final summary
    summary_df = summary_df.drop_duplicates(subset=['query_id', 'corpus_id'])

    summary_df.to_csv(os.path.join(full_folder, "summary.tsv"), sep="\t", index=False)

    print(f"\n📊 COMBINED FULL DATASET stats:")
    print(f" - Total Unique Queries: {len(combined_queries)}")
    print(f" - Total Unique Corpus passages: {len(combined_corpus)}")
    print(f" - Total Positives: {len(combined_positives)}")
    print(f" - Total Unique Qrels: {len(combined_qrels)}")
    print(f" - Final Summary records: {len(summary_df)}")

# ============================================================
# Main
# ============================================================

def regenerate_all_webquestions():
    print("🚀 REGENERATING ALL WebQuestions splits (with DBpedia linking + notes)")

    cache = load_cache()

    splits = {
        'train': 'web_questions_train.json',
        'val': 'web_questions_validation.json',
        'test': 'web_questions_test.json'
    }

    all_data = {}  # Store data from all splits

    # Define unique offsets for each split to avoid ID conflicts
    split_offsets = {'train': 0, 'val': 1, 'test': 2}

    # First process all splits and store their data with unique offsets
    for split_name, input_file in splits.items():
        output_dir = os.path.dirname(input_file) or '.'
        offset = split_offsets[split_name]

        # Process this split with unique offset and store the data
        queries, corpus, positives, qrels = process_jsonl(input_file, output_dir, cache, offset)
        all_data[split_name] = (queries, corpus, positives, qrels)

        # Save individual split
        save_individual_splits(output_dir, split_name, queries, corpus, positives, qrels)

    # Now save combined full dataset
    output_dir = os.path.dirname(list(splits.values())[0]) or '.'
    save_combined_full_dataset(output_dir, all_data)

    save_cache(cache)
    print("\n🎉 All WebQuestions datasets regenerated successfully!")
    print("   - Individual splits saved in train/, val/, test/ folders")
    print("   - Combined dataset saved in full/ folder")

if __name__ == "__main__":
    regenerate_all_webquestions()