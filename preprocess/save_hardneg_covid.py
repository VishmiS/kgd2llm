# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/preprocess/save_hardneg_covid.py

import sys
import pathlib

project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import os
import sys
import csv
import torch
import pathlib
import warnings
import numpy as np
from enum import Enum
from typing import List, Union
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm, trange
from utils.common_utils import set_seed, write_pickle, load_pickle
import faiss

warnings.filterwarnings('ignore')

# Increase CSV field size limit
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

# Set your config here
config = {
    'sets': ['train','val'],
    'base_dir': 'dataset/covid',
    'output_base': 'outputs/neg_covid',
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'K': 8,
    'max_seq_len': 250,
    'use_sample': False,
    'sample_limit': 10000,
    'encoder_type': 'MEAN'
}


def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


class EncoderType(Enum):
    CLS = 2
    MEAN = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError(f"Invalid encoder type: {s}")


class BaseBertModel:
    def __init__(self, model_name_or_path, max_seq_length=512, encoder_type='MEAN'):  # ✅ CHANGED TO MEAN
        self.encoder_type = EncoderType.from_string(encoder_type)
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.plm_model = AutoModel.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plm_model.to(self.device)

        # ✅ ADD: Freeze some layers to prevent over-smoothing
        if hasattr(self.plm_model, 'encoder'):
            # Freeze first 6 layers of BERT
            for layer in self.plm_model.encoder.layer[:6]:
                for param in layer.parameters():
                    param.requires_grad = False

    def get_sentence_embeddings(self, input_ids, attention_mask):
        output = self.plm_model(input_ids=input_ids, attention_mask=attention_mask)

        if self.encoder_type == EncoderType.CLS:
            embeddings = output.last_hidden_state[:, 0]
        elif self.encoder_type == EncoderType.MEAN:
            # ✅ IMPROVED: Weighted mean pooling (better than simple mean)
            token_embeddings = output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            # ✅ Use inverse frequency weighting to reduce common token dominance
            seq_length = token_embeddings.shape[1]
            position_weights = 1.0 / (torch.arange(1, seq_length + 1).float().to(self.device))
            position_weights = position_weights.unsqueeze(0).unsqueeze(-1).expand_as(token_embeddings)

            weighted_embeddings = token_embeddings * position_weights * input_mask_expanded
            sum_embeddings = torch.sum(weighted_embeddings, 1)
            sum_mask = torch.clamp((input_mask_expanded * position_weights).sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

        # ✅ ADD: Layer normalization before final normalization
        embeddings = torch.nn.functional.layer_norm(embeddings, (embeddings.size(-1),))

        return embeddings

    def batch_to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def encode(self, sentences: Union[str, List[str]], batch_size=32, max_seq_length=None):
        self.plm_model.eval()
        if isinstance(sentences, str):
            sentences = [sentences]
        max_seq_length = max_seq_length or self.max_seq_length
        embeddings = []

        for start in trange(0, len(sentences), batch_size, desc="Encoding", leave=False, dynamic_ncols=True):
            batch = sentences[start:start + batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(batch, padding=True, truncation=True,
                                        max_length=max_seq_length, return_tensors='pt')
                inputs = self.batch_to_device(inputs)
                embs = self.get_sentence_embeddings(inputs["input_ids"], inputs["attention_mask"])

                # ✅ ADD: Diversity enhancement through random projection
                if embs.size(0) > 1:  # Only for batches > 1
                    # Small random noise to break symmetry
                    noise = torch.randn_like(embs) * 0.01
                    embs = embs + noise

                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                embeddings.extend(embs.cpu().numpy())

        # ✅ ADD: Post-processing to ensure diversity
        embeddings_array = np.array(embeddings)
        if len(embeddings_array) > 1:
            # Apply whitening if too similar
            similarity_matrix = np.dot(embeddings_array, embeddings_array.T)
            avg_similarity = np.mean(similarity_matrix - np.eye(len(embeddings_array)))
            if avg_similarity > 0.5:
                print(f"⚠️  High embedding similarity detected: {avg_similarity:.4f}, applying whitening")
                # Simple whitening
                embeddings_array = embeddings_array - np.mean(embeddings_array, axis=0)
                embeddings_array = embeddings_array / (np.std(embeddings_array, axis=0) + 1e-8)
                embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)

        return embeddings_array


def save_corpus(model, tsv_path, output_dir):
    ensure_dir(output_dir)
    corpus_set = set()

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            text = row['text'].strip()
            if text:
                corpus_set.add(text[:320])
            if config.get('use_sample', False) and len(corpus_set) >= config.get('sample_limit', float('inf')):
                break

    corpus = list(corpus_set)
    corpus_id_map = {text: idx for idx, text in enumerate(corpus)}

    embeddings = model.encode(corpus, batch_size=512, max_seq_length=config['max_seq_len'])

    write_pickle(corpus_id_map, os.path.join(output_dir, 'corpus_id_map.pkl'))
    write_pickle(corpus, os.path.join(output_dir, 'corpus_texts.pkl'))
    write_pickle(embeddings, os.path.join(output_dir, 'corpus_embeddings.pkl'))

    print(f"Corpus saved at {output_dir} ({len(corpus)} entries)")


def create_query_variations(original_query, num_variations=7):
    """Create semantic variations of queries to increase dataset size"""
    variations = []

    # Common paraphrasing patterns for COVID queries
    covid_paraphrases = [
        "What is known about {query}",
        "Information regarding {query}",
        "Details about {query}",
        "Research on {query}",
        "Studies about {query}",
        "Findings related to {query}",
        "Data concerning {query}"
    ]

    # Remove any existing prefix patterns to avoid duplication
    base_query = original_query
    prefixes = ["What is known about ", "Information regarding ", "Details about ",
                "Research on ", "Studies about ", "Findings related to ", "Data concerning "]

    for prefix in prefixes:
        if base_query.startswith(prefix):
            base_query = base_query[len(prefix):].strip()
            break

    # Create variations
    for i in range(min(num_variations, len(covid_paraphrases))):
        variation = covid_paraphrases[i].format(query=base_query)
        variations.append(variation)

    return variations


def save_queries_and_hard_negatives(model, positive_path, corpus_embeddings_path,
                                    corpus_texts_path, K, output_path):
    """Load original TREC-COVID queries instead of duplicated positives.tsv"""

    # 🔥 NEW: Load from original query file instead of positives.tsv
    queries_dir = os.path.dirname(positive_path)
    queries_file = os.path.join(queries_dir, 'queries.tsv')

    queries = []
    pos_dict = defaultdict(set)

    # Try to load from original query file first
    if os.path.exists(queries_file):
        print("📁 Loading original queries from queries.tsv...")
        with open(queries_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                q = row.get('text', '').strip() or row.get('query', '').strip()
                if q:
                    queries.append(q)
                    pos_dict[q] = set()  # Initialize empty
    else:
        # Fallback: extract unique queries from positives.tsv
        print("⚠️  queries.tsv not found, extracting unique queries from positives.tsv...")
        seen_queries = set()
        with open(positive_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                q = row.get('sentence1', '').strip()
                a = row.get('sentence2', '').strip()
                if q and a:
                    if q not in seen_queries:
                        queries.append(q)
                        seen_queries.add(q)
                    pos_dict[q].add(a)

    print(f"✅ Unique queries loaded: {len(queries)}")

    # 🔥 CRITICAL FIX: Load positives from positives.tsv to populate pos_dict
    print("📁 Loading positive passages from positives.tsv...")
    with open(positive_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        positives_loaded = 0
        for row in reader:
            q = row.get('sentence1', '').strip()
            a = row.get('sentence2', '').strip()
            if q and a and q in pos_dict:  # Only add if query exists in our list
                pos_dict[q].add(a)
                positives_loaded += 1

    print(f"✅ Loaded {positives_loaded} positive passages for {len(queries)} queries")

    # Print detailed stats about positives
    print("\n📊 Positive Passage Statistics:")
    for i, q in enumerate(queries):
        print(f"   Query {i + 1}: '{q[:60]}...' -> {len(pos_dict[q])} positive passages")
        if i >= 4:  # Show first 5 queries
            print(f"   ... and {len(queries) - 5} more queries")
            break

    # 🔥 Apply query variations to the unique queries
    print("\n🔄 Creating query variations...")
    augmented_queries = []
    augmented_pos_dict = defaultdict(set)

    for i, original_query in enumerate(queries):
        # Keep the original query
        augmented_queries.append(original_query)
        augmented_pos_dict[original_query] = pos_dict[original_query].copy()

        # Create variations (this will 8x your dataset size)
        variations = create_query_variations(original_query, num_variations=7)

        for variation in variations:
            augmented_queries.append(variation)
            # Use the same positive answers for variations
            augmented_pos_dict[variation] = pos_dict[original_query].copy()

    queries = augmented_queries
    pos_dict = augmented_pos_dict

    print(f"✅ After augmentation: {len(queries)} queries (from {len(queries) // 8} unique originals)")

    # Rest of your existing code for encoding and FAISS search...
    corpus_texts = load_pickle(corpus_texts_path)
    corpus_embeddings = load_pickle(corpus_embeddings_path)
    print(f"📊 Corpus size: {len(corpus_texts)} documents")

    query_embeddings = model.encode(queries, batch_size=512, max_seq_length=100)
    write_pickle(query_embeddings, os.path.join(os.path.dirname(output_path), 'query_embeddings.pkl'))

    corpus_embs = corpus_embeddings.astype('float32')
    query_embs = query_embeddings.astype('float32')

    # Simple GPU version
    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, corpus_embs.shape[1])
    else:
        index = faiss.IndexFlatIP(corpus_embs.shape[1])

    index.add(corpus_embs)

    # Search for candidates
    search_k = min(K * 10, len(corpus_texts))
    print(f"\n🔍 Searching top-{search_k} candidates per query to ensure {K} quality negatives")

    D, I = index.search(query_embs, search_k)

    # Clean up
    if torch.cuda.is_available():
        del index
        res.noTempMemory()

    # 🔥 FIXED: Initialize stats properly and add detailed debugging
    result = defaultdict(list)
    stats = {'full_k': 0, 'partial_k': 0, 'no_candidates': 0}
    query_details = []  # Store detailed info for each query

    print("\n📈 Processing each query:")
    for i, (neighbors, scores) in enumerate(
            tqdm(zip(I, D), total=len(queries), desc="Mining hard negatives", ncols=100)):
        q = queries[i]
        seen = set()
        valid_negatives = []
        candidate_pool = []

        # Detailed candidate collection
        candidates_before_filter = 0
        candidates_after_filter = 0

        for idx, score in zip(neighbors, scores):
            candidates_before_filter += 1
            candidate = corpus_texts[idx]

            # Skip invalid candidates
            if (not candidate or candidate == q or candidate in pos_dict[q] or
                    candidate in seen or len(candidate.split()) < 3):
                continue

            # 🔥 RELAX THE SIMILARITY THRESHOLD SIGNIFICANTLY
            if score < 0.01:  # Reduced from 0.05 to get more candidates
                continue

            candidate_pool.append((candidate, score))
            seen.add(candidate)
            candidates_after_filter += 1

        # Sort and select top candidates
        candidate_pool.sort(key=lambda x: x[1], reverse=True)
        for candidate, score in candidate_pool:
            if len(valid_negatives) < K:
                valid_negatives.append(candidate)
            else:
                break

        # 🔥 IMPROVED FALLBACK: If not enough candidates, relax criteria further
        fallback_used = False
        if len(valid_negatives) < K:
            fallback_used = True
            # Try again with even lower threshold and relaxed length requirement
            additional_candidates = []
            for idx, score in zip(neighbors, scores):
                candidate = corpus_texts[idx]
                if (candidate and candidate != q and candidate not in pos_dict[q] and
                        candidate not in valid_negatives and candidate not in additional_candidates and
                        len(candidate.split()) >= 2):  # Reduced from 3 to 2
                    if score >= 0.001:  # Very low threshold
                        additional_candidates.append(candidate)
                    if len(valid_negatives) + len(additional_candidates) >= K:
                        break

            valid_negatives.extend(additional_candidates)

        # 🔥 ULTIMATE FALLBACK: If still not enough, use any available documents
        if len(valid_negatives) < K:
            emergency_candidates = []
            for idx, score in zip(neighbors, scores):
                candidate = corpus_texts[idx]
                if (candidate and candidate != q and candidate not in pos_dict[q] and
                        candidate not in valid_negatives and candidate not in emergency_candidates):
                    emergency_candidates.append(candidate)
                    if len(valid_negatives) + len(emergency_candidates) >= K:
                        break

            valid_negatives.extend(emergency_candidates)

        # Store detailed query information
        query_info = {
            'query': q,
            'negatives_found': len(valid_negatives),
            'candidates_before_filter': candidates_before_filter,
            'candidates_after_filter': candidates_after_filter,
            'fallback_used': fallback_used,
            'target_k': K
        }
        query_details.append(query_info)

        # Print detailed info for first 10 queries and any problematic ones
        if i < 10 or len(valid_negatives) < K:
            status = "✅" if len(valid_negatives) == K else "⚠️" if len(valid_negatives) > 0 else "❌"
            print(f"   {status} Query {i}: Found {len(valid_negatives)}/{K} negatives "
                  f"(candidates: {candidates_after_filter}/{candidates_before_filter})"
                  f"{' [FALLBACK]' if fallback_used else ''}")

        # Final validation and stats
        if len(valid_negatives) == K:
            stats['full_k'] += 1
        elif len(valid_negatives) > 0:
            stats['partial_k'] += 1
        else:
            stats['no_candidates'] += 1
            print(f"   ❌ Query {i}: ZERO candidates found!")

        result[q] = valid_negatives[:K]

    # 🔥 COMPREHENSIVE FINAL STATISTICS
    print(f"\n" + "=" * 60)
    print("📊 HARD NEGATIVE GENERATION - COMPREHENSIVE REPORT")
    print("=" * 60)

    # Overall statistics
    print(f"📈 Overall Statistics:")
    print(f"   • Total queries processed: {len(queries)}")
    print(f"   • Queries with full {K} negatives: {stats['full_k']} ({stats['full_k'] / len(queries) * 100:.1f}%)")
    print(f"   • Queries with partial negatives: {stats['partial_k']} ({stats['partial_k'] / len(queries) * 100:.1f}%)")
    print(
        f"   • Queries with NO candidates: {stats['no_candidates']} ({stats['no_candidates'] / len(queries) * 100:.1f}%)")

    # Detailed breakdown
    print(f"\n🔍 Detailed Breakdown:")
    negatives_distribution = defaultdict(int)
    for detail in query_details:
        negatives_distribution[detail['negatives_found']] += 1

    for count, num_queries in sorted(negatives_distribution.items()):
        percentage = (num_queries / len(queries)) * 100
        status = "✅" if count == K else "⚠️" if count > 0 else "❌"
        print(f"   {status} {count:2d} negatives: {num_queries:3d} queries ({percentage:5.1f}%)")

    # Problematic queries
    problematic_queries = [qd for qd in query_details if qd['negatives_found'] < K]
    if problematic_queries:
        print(f"\n🚨 Problematic Queries (less than {K} negatives):")
        for qd in problematic_queries[:10]:  # Show first 10
            print(f"   • '{qd['query'][:50]}...' -> {qd['negatives_found']}/{K} negatives")
        if len(problematic_queries) > 10:
            print(f"   ... and {len(problematic_queries) - 10} more")

    # Success metrics
    total_negatives_generated = sum(len(negs) for negs in result.values())
    max_possible_negatives = len(queries) * K
    coverage_rate = (total_negatives_generated / max_possible_negatives) * 100

    print(f"\n🎯 Success Metrics:")
    print(f"   • Total negatives generated: {total_negatives_generated}/{max_possible_negatives}")
    print(f"   • Coverage rate: {coverage_rate:.1f}%")
    print(f"   • Average negatives per query: {total_negatives_generated / len(queries):.1f}")

    write_pickle(result, output_path)
    print(f"\n💾 Hard negatives saved to: {output_path}")
    print("=" * 60)


def process_dataset(split: str, model: BaseBertModel):
    base_dir = config['base_dir']
    output_dir = os.path.join(config['output_base'], split)

    input_dir = os.path.join(base_dir, split)
    ensure_dir(input_dir)
    ensure_dir(output_dir)

    corpus_path = os.path.join(input_dir, 'corpus.tsv')
    positive_path = os.path.join(input_dir, 'positives.tsv')

    for f in [corpus_path, positive_path]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Missing expected file: {f}")
        else:
            print(f"Found input file: {f}")

    save_corpus(model, corpus_path, output_dir)

    save_queries_and_hard_negatives(
        model=model,
        positive_path=positive_path,
        corpus_embeddings_path=os.path.join(output_dir, 'corpus_embeddings.pkl'),
        corpus_texts_path=os.path.join(output_dir, 'corpus_texts.pkl'),
        K=config['K'],
        output_path=os.path.join(output_dir, 'query_hard_negatives.pkl')
    )


def main():
    set_seed(42)
    model = BaseBertModel(config['model_name'], config['max_seq_len'], encoder_type=config.get('encoder_type', 'MEAN'))

    for split in config['sets']:
        print(f"\n--- Processing '{split}' split ---")
        process_dataset(split, model)


if __name__ == '__main__':
    main()
