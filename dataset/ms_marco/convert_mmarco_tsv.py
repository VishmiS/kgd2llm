import os
import json
import csv
import random
import re
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# Utility Functions
# ============================================================

def normalize_text(text: str) -> str:
    """
    Ultra-conservative text normalization.
    Only removes URLs and normalizes whitespace, keeps everything else.
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # ONLY normalize whitespace - keep ALL other characters
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def save_tsv(data, path, fieldnames):
    """Save data as a TSV file with given headers."""
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Saved: {path} ({len(data)} records)")


# ============================================================
# Data Loading and Analysis
# ============================================================

def load_ms_marco_json(json_path):
    """Load the raw MS MARCO JSON file."""
    print(f"Loading dataset: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    passages_data = data.get('passages', {})
    queries_data = data.get('query', {})

    print(f"Dataset Overview:")
    print(f" - Passages: {len(passages_data)} queries with passages")
    print(f" - Queries:  {len(queries_data)}")
    return passages_data, queries_data


def analyze_dataset(passages_data, queries_data):
    """Analyze the dataset to get unique queries, passages, and qrels."""
    print(f"Analyzing dataset statistics...")

    # Get all unique queries
    unique_queries = set()
    for qid, qtext in queries_data.items():
        if qtext and qid in passages_data:
            unique_queries.add(qid)

    # Get all unique passages and qrels
    unique_passages = set()
    all_qrels = []

    for qid, plist in passages_data.items():
        if qid not in unique_queries:
            continue

        for idx, p in enumerate(plist):
            pid = f"{qid}_{idx}"
            ptext = p.get('passage_text', '')
            if ptext:  # Only count non-empty passages
                unique_passages.add(pid)

                if p.get("is_selected", 0) == 1:
                    all_qrels.append({
                        'query_id': qid,
                        'corpus_id': pid,
                        'rel': 1
                    })

    print(f"Original Dataset Statistics:")
    print(f" - Unique queries:   {len(unique_queries)}")
    print(f" - Unique passages:  {len(unique_passages)}")
    print(f" - Total qrels:      {len(all_qrels)}")

    return unique_queries, unique_passages, all_qrels


def extract_30_percent_subset(passages_data, queries_data, all_qrels, fraction=0.3):
    """Extract 30% of queries and ALL their passages, then sample qrels from those."""
    print(f"Extracting {fraction * 100:.0f}% subset of queries with ALL their passages...")

    # Get all unique query IDs that have qrels
    queries_with_qrels = set(qrel['query_id'] for qrel in all_qrels)

    # Sample 30% of queries (not qrels)
    subset_query_ids = set(random.sample(list(queries_with_qrels), int(len(queries_with_qrels) * fraction)))

    # Get ALL passages for the sampled queries (not just the ones in qrels)
    subset_passages = {}
    subset_queries = {}
    all_passages_from_sampled_queries = set()

    for qid in subset_query_ids:
        if qid in queries_data:
            subset_queries[qid] = queries_data[qid]
        if qid in passages_data:
            # Include ALL passages for these queries, not just the ones in qrels
            subset_passages[qid] = passages_data[qid]
            for idx, p in enumerate(passages_data[qid]):
                pid = f"{qid}_{idx}"
                all_passages_from_sampled_queries.add(pid)

    # Now sample qrels only from our subset of queries
    subset_qrels = [qrel for qrel in all_qrels if qrel['query_id'] in subset_query_ids]

    print(f"30% Subset Statistics:")
    print(f" - Sampled queries:   {len(subset_query_ids)}")
    print(f" - All passages from sampled queries: {len(all_passages_from_sampled_queries)}")
    print(f" - Qrels from sampled queries: {len(subset_qrels)}")

    return subset_passages, subset_queries, subset_qrels, subset_query_ids, all_passages_from_sampled_queries


# ============================================================
# Corpus Trimming
# ============================================================

def trim_corpus_passages(passages_data, max_tokens=300):
    """Trim corpus passages exceeding max_tokens and count how many were trimmed."""
    trimmed_count = 0
    trimmed_passages = {}
    for qid, plist in passages_data.items():
        trimmed_list = []
        for p in plist:
            ptext = p.get('passage_text', '')
            # Apply minimal normalization just for length checking
            temp_text = normalize_text(ptext)
            tokens = temp_text.split()
            if len(tokens) > max_tokens:
                # Trim the original text, not normalized one
                original_tokens = ptext.split()
                ptext = " ".join(original_tokens[:max_tokens])
                trimmed_count += 1
            trimmed_list.append({**p, "passage_text": ptext})
        trimmed_passages[qid] = trimmed_list
    print(f"Trimmed {trimmed_count} passages exceeding {max_tokens} tokens.")
    return trimmed_passages


# ============================================================
# Data Construction - UPDATED VERSION
# ============================================================

def build_datasets(passages_data, queries_data, subset_qrels, subset_corpus_ids):
    """Build corpus, queries, qrels, positives lists from the 30% subset.
    Includes ALL passages from the sampled queries, not just those in qrels.
    ROBUST VERSION: Handles normalization issues gracefully.
    """
    print(f"Building datasets from 30% subset...")

    corpus, queries, qrels, positives = [], [], [], []
    seen_passages, seen_queries = set(), set()

    # Create a lookup for our sampled qrels
    qrel_lookup = set((qrel['query_id'], qrel['corpus_id']) for qrel in subset_qrels)

    # First pass: Build ALL queries and ALL passages from our sampled queries
    query_id_to_text = {}
    passage_id_to_text = {}

    # Track stats for debugging
    stats = {
        'queries_processed': 0,
        'queries_skipped': 0,
        'passages_processed': 0,
        'passages_skipped': 0
    }

    # Process ALL queries that are in our sampled subset (not just those with qrels)
    for qid in tqdm(subset_corpus_ids, desc="Processing ALL queries and passages"):
        if qid not in queries_data or qid not in passages_data:
            stats['queries_skipped'] += 1
            continue

        # Process query - VERY LENIENT
        qtext = queries_data.get(qid, "")
        if not qtext or not isinstance(qtext, str):
            stats['queries_skipped'] += 1
            continue

        # Minimal normalization for queries
        normalized_qtext = normalize_text(qtext)
        if not normalized_qtext or normalized_qtext.isspace():
            # If normalization fails, use original text
            normalized_qtext = qtext.lower().strip()

        # Store query (allow duplicates to preserve qrels)
        queries.append({'id': qid, 'query': normalized_qtext})
        query_id_to_text[qid] = normalized_qtext
        stats['queries_processed'] += 1

        # Process ALL passages for this query (not just those in qrels)
        for idx, p in enumerate(passages_data[qid]):
            pid = f"{qid}_{idx}"
            ptext = p.get('passage_text', '')

            if not ptext or not isinstance(ptext, str):
                stats['passages_skipped'] += 1
                continue

            # Minimal normalization for passages
            normalized_ptext = normalize_text(ptext)
            if not normalized_ptext or normalized_ptext.isspace():
                # If normalization fails, use original text
                normalized_ptext = ptext.lower().strip()

            # Store passage (allow duplicates to preserve qrels)
            corpus.append({'id': pid, 'text': normalized_ptext})
            passage_id_to_text[pid] = normalized_ptext
            stats['passages_processed'] += 1

    print(f"Processing Statistics:")
    print(f" - Queries processed: {stats['queries_processed']}")
    print(f" - Queries skipped: {stats['queries_skipped']}")
    print(f" - Passages processed: {stats['passages_processed']}")
    print(f" - Passages skipped: {stats['passages_skipped']}")

    # Second pass: Build qrels and positives only for relevant passages
    successful_qrels = 0
    lost_qrels = 0

    for qrel in tqdm(subset_qrels, desc="Building final qrels and positives"):
        qid = qrel['query_id']
        pid = qrel['corpus_id']

        # Only create qrel if both query and passage exist
        if qid in query_id_to_text and pid in passage_id_to_text:
            qrels.append({'query_id': qid, 'corpus_id': pid, 'rel': 1})
            positives.append({
                'sentence1': query_id_to_text[qid],
                'sentence2': passage_id_to_text[pid],
                'query_id': qid,
                'passage_id': pid
            })
            successful_qrels += 1
        else:
            lost_qrels += 1
            # Only show first few lost qrels to avoid spam
            if lost_qrels <= 10:
                if qid not in query_id_to_text:
                    print(f"Lost qrel - missing query: {qid}")
                if pid not in passage_id_to_text:
                    print(f"Lost qrel - missing passage: {pid} for query: {qid}")

    print(f"Final 30% Subset Statistics:")
    print(f" - Corpus:    {len(corpus)}")
    print(f" - Queries:   {len(queries)}")
    print(f" - Qrels:     {len(qrels)}")
    print(f" - Positives: {len(positives)}")

    # Debug info
    original_qrels_count = len(subset_qrels)
    final_qrels_count = len(qrels)
    if final_qrels_count < original_qrels_count:
        print(f"Warning: Lost {original_qrels_count - final_qrels_count} qrels ({(final_qrels_count / original_qrels_count) * 100:.1f}% retention)")
    else:
        print(f"Retained {final_qrels_count} qrels (100% retention)")

    # CRITICAL: If we have no data, create minimal dummy data to avoid crashes
    if len(corpus) == 0 or len(queries) == 0:
        print("WARNING: No data generated! Creating minimal dummy data...")
        # Create minimal dummy entries to avoid crashes
        if len(corpus) == 0:
            corpus.append({'id': 'dummy_passage', 'text': 'sample passage text'})
        if len(queries) == 0:
            queries.append({'id': 'dummy_query', 'query': 'sample query'})
        if len(qrels) == 0 and len(queries) > 0 and len(corpus) > 0:
            qrels.append({'query_id': queries[0]['id'], 'corpus_id': corpus[0]['id'], 'rel': 1})

    return corpus, queries, qrels, positives


# ============================================================
# Split Logic - UPDATED VERSION
# ============================================================

def split_full30_fixed(full30_dir="./full30", split_dir_base=".", test_queries=5000, val_queries=5000):
    """
    Split full30 dataset into train/val/test with 5000 queries each for val and test.
    ALL splits share the SAME corpus from the original 30% subset.
    ROBUST VERSION: Handles edge cases gracefully.
    """
    print(f"Splitting dataset with {val_queries} val and {test_queries} test queries...")

    # Load the full30 TSVs
    try:
        queries = pd.read_csv(os.path.join(full30_dir, 'queries.tsv'), sep='\t')
        corpus = pd.read_csv(os.path.join(full30_dir, 'corpus.tsv'), sep='\t')
        qrels = pd.read_csv(os.path.join(full30_dir, 'qrels.tsv'), sep='\t')
        positives = pd.read_csv(os.path.join(full30_dir, 'positives.tsv'), sep='\t')
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Use positives to regenerate qrels for consistency
    regenerated_qrels = positives[['query_id', 'passage_id']].copy()
    regenerated_qrels['rel'] = 1
    regenerated_qrels.columns = ['query_id', 'corpus_id', 'rel']

    # Use the regenerated qrels
    qrels = regenerated_qrels

    # Get unique query_ids from the regenerated qrels
    all_query_ids = qrels['query_id'].unique()
    total_queries = len(all_query_ids)

    print(f"Qrels consistency check:")
    print(f" - Regenerated qrels count: {len(qrels)}")
    print(f" - Positives count: {len(positives)}")
    print(f"Total unique queries in qrels: {total_queries}")

    # ADJUST SPLIT SIZES BASED ON AVAILABLE DATA
    available_queries = total_queries
    adjusted_test_queries = min(test_queries, available_queries // 3)
    adjusted_val_queries = min(val_queries, (available_queries - adjusted_test_queries) // 2)

    print(f"Adjusted split sizes (based on available data):")
    print(f" - Test:  {adjusted_test_queries} queries")
    print(f" - Val:   {adjusted_val_queries} queries")
    print(f" - Train: {available_queries - adjusted_test_queries - adjusted_val_queries} queries")
    print(f" - Corpus: {len(corpus)} passages (SAME for all splits)")

    # Check if we have enough data for splitting
    if available_queries < 3:
        print("Not enough queries for splitting. Creating single 'all' split instead.")
        splits = {'all': set(all_query_ids)}
    else:
        # Split the data
        if adjusted_test_queries > 0:
            train_val_ids, test_ids = train_test_split(
                all_query_ids,
                test_size=adjusted_test_queries,
                random_state=42
            )
        else:
            train_val_ids, test_ids = all_query_ids, []

        if adjusted_val_queries > 0 and len(train_val_ids) > adjusted_val_queries:
            val_proportion = adjusted_val_queries / len(train_val_ids)
            train_ids, val_ids = train_test_split(
                train_val_ids,
                test_size=val_proportion,
                random_state=42
            )
        else:
            train_ids, val_ids = train_val_ids, []

        splits = {
            'train': set(train_ids),
            'val': set(val_ids) if len(val_ids) > 0 else set(),
            'test': set(test_ids) if len(test_ids) > 0 else set()
        }

    # Verify split sizes
    print(f"Split Verification:")
    for split_name, split_query_ids in splits.items():
        print(f" - {split_name.capitalize()} queries: {len(split_query_ids)}")
    print(f" - Shared corpus: {len(corpus)} passages")

    # Process each split
    for split_name, split_query_ids in splits.items():
        if len(split_query_ids) == 0:
            print(f"Skipping {split_name} split (no queries)")
            continue

        print(f"Processing {split_name} split...")

        # Filter queries that are in this split
        split_queries = queries[queries['id'].isin(split_query_ids)]

        # Get the actual query IDs that exist in our queries file
        actual_split_query_ids = set(split_queries['id'])

        # Filter qrels for this split
        split_qrels = qrels[
            qrels['query_id'].isin(actual_split_query_ids) &
            qrels['corpus_id'].isin(corpus['id'])
            ]

        # Use the SAME corpus for ALL splits
        split_corpus = corpus.copy()

        # Save TSVs
        split_dir = os.path.join(split_dir_base, split_name)
        os.makedirs(split_dir, exist_ok=True)

        split_queries.to_csv(os.path.join(split_dir, 'queries.tsv'), sep='\t', index=False)
        split_corpus.to_csv(os.path.join(split_dir, 'corpus.tsv'), sep='\t', index=False)
        split_qrels.to_csv(os.path.join(split_dir, 'qrels.tsv'), sep='\t', index=False)

        # Generate positives
        split_positives = pd.merge(
            split_qrels[split_qrels['rel'] == 1],
            split_queries,
            left_on='query_id',
            right_on='id',
            how='inner'
        ).merge(
            split_corpus,
            left_on='corpus_id',
            right_on='id',
            how='inner',
            suffixes=('_query', '_corpus')
        )

        if len(split_positives) > 0:
            split_positives = split_positives[['query', 'text']]
            split_positives.columns = ['sentence1', 'sentence2']
            split_positives.to_csv(os.path.join(split_dir, 'positives.tsv'), sep='\t', index=False)

        # Generate summary
        summary = pd.merge(
            split_qrels,
            split_queries,
            left_on='query_id',
            right_on='id',
            how='inner'
        ).merge(
            split_corpus,
            left_on='corpus_id',
            right_on='id',
            how='inner',
            suffixes=('_query', '_corpus')
        )[['query_id', 'query', 'corpus_id', 'text', 'rel']]
        summary.columns = ['query_id', 'sentence1', 'corpus_id', 'sentence2', 'rel']
        summary.to_csv(os.path.join(split_dir, 'summary.tsv'), sep='\t', index=False)

        print(f"Saved {split_name} split:")
        print(f"   - Queries: {len(split_queries)}")
        print(f"   - Corpus:  {len(split_corpus)}")
        print(f"   - Qrels:   {len(split_qrels)}")
        print(f"   - Positives: {len(split_positives)}")


# ============================================================
# Main Processing Pipeline
# ============================================================

def process_ms_marco(json_path, subset_fraction=0.3, max_corpus_tokens=300):
    """Main processing pipeline with proper 30% qrels sampling."""

    # Step 1: Load data
    passages_data, queries_data = load_ms_marco_json(json_path)

    # Step 2: Analyze original dataset
    unique_queries, unique_passages, all_qrels = analyze_dataset(passages_data, queries_data)

    # Step 3: Extract 30% subset based on qrels
    subset_passages, subset_queries, subset_qrels, subset_query_ids, subset_corpus_ids = extract_30_percent_subset(
        passages_data, queries_data, all_qrels, fraction=subset_fraction
    )

    # Step 4: Trim corpus passages
    subset_passages = trim_corpus_passages(subset_passages, max_tokens=max_corpus_tokens)

    # Step 5: Build datasets from the 30% subset
    corpus, queries, qrels, positives = build_datasets(subset_passages, subset_queries, subset_qrels, subset_query_ids)

    # Step 6: Save full 30% subset
    os.makedirs("./full30", exist_ok=True)
    save_tsv(corpus, './full30/corpus.tsv', ['id', 'text'])
    save_tsv(queries, './full30/queries.tsv', ['id', 'query'])
    save_tsv(qrels, './full30/qrels.tsv', ['query_id', 'corpus_id', 'rel'])
    save_tsv(positives, './full30/positives.tsv', ['sentence1', 'sentence2', 'query_id', 'passage_id'])

    # Step 7: Create summary file for full30
    summary_data = []
    for pos in positives:
        summary_data.append({
            'query_id': pos['query_id'],
            'sentence1': pos['sentence1'],
            'corpus_id': pos['passage_id'],
            'sentence2': pos['sentence2'],
            'rel': 1
        })
    save_tsv(summary_data, './full30/summary.tsv', ['query_id', 'sentence1', 'corpus_id', 'sentence2', 'rel'])

    # Step 8: Split into train/val/test
    split_full30_fixed(full30_dir="./full30", split_dir_base=".", test_queries=5000, val_queries=5000)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    process_ms_marco('original_downloads/train_v2.1.json', subset_fraction=0.3, max_corpus_tokens=300)