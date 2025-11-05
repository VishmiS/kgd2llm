# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/preprocess/save_hardneg_webq.py

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

# Set your config here - REDUCED K for better quality
config = {
    'sets': ['train', 'val'],
    'base_dir': 'dataset/web_questions',
    'output_base': 'outputs/neg_web_questions',
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'K': 5,  # 🔥 REDUCED from 10 to 5 for higher quality negatives
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
    def __init__(self, model_name_or_path, max_seq_length=512, encoder_type='MEAN'):
        self.encoder_type = EncoderType.from_string(encoder_type)
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.plm_model = AutoModel.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plm_model.to(self.device)

        # Freeze some layers to prevent over-smoothing
        if hasattr(self.plm_model, 'encoder'):
            for layer in self.plm_model.encoder.layer[:6]:
                for param in layer.parameters():
                    param.requires_grad = False

    def get_sentence_embeddings(self, input_ids, attention_mask):
        output = self.plm_model(input_ids=input_ids, attention_mask=attention_mask)

        if self.encoder_type == EncoderType.CLS:
            embeddings = output.last_hidden_state[:, 0]
        elif self.encoder_type == EncoderType.MEAN:
            token_embeddings = output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            seq_length = token_embeddings.shape[1]
            position_weights = 1.0 / (torch.arange(1, seq_length + 1).float().to(self.device))
            position_weights = position_weights.unsqueeze(0).unsqueeze(-1).expand_as(token_embeddings)

            weighted_embeddings = token_embeddings * position_weights * input_mask_expanded
            sum_embeddings = torch.sum(weighted_embeddings, 1)
            sum_mask = torch.clamp((input_mask_expanded * position_weights).sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask

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

                if embs.size(0) > 1:
                    noise = torch.randn_like(embs) * 0.01
                    embs = embs + noise

                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                embeddings.extend(embs.cpu().numpy())

        embeddings_array = np.array(embeddings)
        if len(embeddings_array) > 1:
            similarity_matrix = np.dot(embeddings_array, embeddings_array.T)
            avg_similarity = np.mean(similarity_matrix - np.eye(len(embeddings_array)))
            if avg_similarity > 0.5:
                print(f"⚠️  High embedding similarity detected: {avg_similarity:.4f}, applying whitening")
                embeddings_array = embeddings_array - np.mean(embeddings_array, axis=0)
                embeddings_array = embeddings_array / (np.std(embeddings_array, axis=0) + 1e-8)
                embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)

        return embeddings_array


def save_corpus(model, tsv_path, output_dir):
    ensure_dir(output_dir)
    corpus_texts = []
    corpus_ids = []
    corpus_data = []

    print(f"📁 Loading corpus from: {tsv_path}")

    # First, let's inspect the file structure more carefully
    print("🔍 Detailed file inspection...")

    with open(tsv_path, 'r', encoding='utf-8') as f:
        # Read first few lines to understand the structure
        first_lines = []
        for i in range(5):
            line = f.readline().strip()
            if line:
                first_lines.append(line)
                print(f"📖 Line {i + 1}: {line[:200]}...")

        # Reset to beginning
        f.seek(0)

        # Try different parsing strategies
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        row_count = 0
        valid_rows = 0

        for row in reader:
            row_count += 1

            # 🔥 FIXED: Handle different possible column structures
            if len(row) >= 2:
                # Strategy 1: Assume first column is ID, second is text
                corpus_id = row[0].strip() if len(row) > 0 else f'row_{row_count}'
                text = row[1].strip() if len(row) > 1 else ''

                # Strategy 2: If that doesn't work, try to find the longest field as text
                if not text and len(row) > 1:
                    # Find the field with most words as text
                    text_candidates = [(i, field.strip()) for i, field in enumerate(row) if field.strip()]
                    if text_candidates:
                        # Prefer fields that are not just numbers
                        non_numeric = [(i, t) for i, t in text_candidates if not t.replace('.', '').isdigit()]
                        if non_numeric:
                            # Use the longest non-numeric field as text
                            text_idx, text = max(non_numeric, key=lambda x: len(x[1]))
                            corpus_id = f'row_{row_count}_col{text_idx}'
                        else:
                            text_idx, text = max(text_candidates, key=lambda x: len(x[1]))
                            corpus_id = f'row_{row_count}_col{text_idx}'

                if text:
                    corpus_texts.append(text)
                    corpus_ids.append(corpus_id)
                    corpus_data.append({
                        'corpus_id': corpus_id,
                        'text': text,
                        'full_row': row  # Store full row for debugging
                    })
                    valid_rows += 1

                    if valid_rows <= 3:  # Show first 3 valid entries
                        print(f"✅ Valid entry {valid_rows}: ID='{corpus_id}', Text='{text[:100]}...'")

            if config.get('use_sample', False) and len(corpus_texts) >= config.get('sample_limit', float('inf')):
                break

    print(f"\n📊 CORPUS LOADING SUMMARY:")
    print(f"   • Total rows processed: {row_count}")
    print(f"   • Valid rows with text: {valid_rows}")
    print(f"   • Invalid rows: {row_count - valid_rows}")

    if len(corpus_texts) == 0:
        print("❌ No valid text entries found. First 5 rows were:")
        for i, line in enumerate(first_lines):
            print(f"   Row {i + 1}: {line}")
        raise ValueError("❌ No valid text entries found in corpus file!")

    corpus_id_map = {corpus_id: idx for idx, corpus_id in enumerate(corpus_ids)}

    print(f"🔄 Encoding {len(corpus_texts)} documents...")
    embeddings = model.encode(corpus_texts, batch_size=512, max_seq_length=config['max_seq_len'])

    write_pickle(corpus_id_map, os.path.join(output_dir, 'corpus_id_map.pkl'))
    write_pickle(corpus_texts, os.path.join(output_dir, 'corpus_texts.pkl'))
    write_pickle(corpus_ids, os.path.join(output_dir, 'corpus_ids.pkl'))
    write_pickle(embeddings, os.path.join(output_dir, 'corpus_embeddings.pkl'))
    write_pickle(corpus_data, os.path.join(output_dir, 'corpus_full_data.pkl'))

    print(f"💾 Corpus saved at {output_dir} ({len(corpus_texts)} entries)")
    print(f"   - Sample texts: {corpus_texts[:2]}")
    return corpus_texts, embeddings


def save_queries_and_hard_negatives(model, positive_path, corpus_embeddings_path,
                                    corpus_texts_path, K, output_path):
    """Improved hard negative mining with better filtering strategies"""

    queries = []
    pos_dict = defaultdict(set)

    # Load queries
    queries_dir = os.path.dirname(positive_path)
    queries_file = os.path.join(queries_dir, 'queries.tsv')

    if os.path.exists(queries_file):
        print("📁 Loading original queries from queries.tsv...")
        with open(queries_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                q = row.get('text', '').strip() or row.get('query', '').strip()
                if q:
                    queries.append(q)
                    pos_dict[q] = set()
    else:
        print("⚠️  queries.tsv not found, extracting unique queries from positives.tsv...")
        seen_queries = set()
        with open(positive_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                q_raw = row.get('sentence1', '')
                q = q_raw.strip() if q_raw is not None else ''
                a_raw = row.get('sentence2', '')
                a = a_raw.strip() if a_raw is not None else ''
                if q and a:
                    if q not in seen_queries:
                        queries.append(q)
                        seen_queries.add(q)
                    pos_dict[q].add(a)

    print(f"✅ Unique queries loaded: {len(queries)}")

    # Load positives
    print("📁 Loading positive passages from positives.tsv...")
    with open(positive_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        positives_loaded = 0
        for row in reader:
            q_raw = row.get('sentence1', '')
            q = q_raw.strip() if q_raw is not None else ''
            a_raw = row.get('sentence2', '')
            a = a_raw.strip() if a_raw is not None else ''
            if q and a and q in pos_dict:
                pos_dict[q].add(a)
                positives_loaded += 1

    print(f"✅ Loaded {positives_loaded} positive passages for {len(queries)} queries")

    # Load corpus
    corpus_texts = load_pickle(corpus_texts_path)
    corpus_embeddings = load_pickle(corpus_embeddings_path)
    print(f"📊 Corpus size: {len(corpus_texts)} documents")

    # Encode queries
    query_embeddings = model.encode(queries, batch_size=512, max_seq_length=100)
    write_pickle(query_embeddings, os.path.join(os.path.dirname(output_path), 'query_embeddings.pkl'))

    # FAISS setup
    corpus_embs = corpus_embeddings.astype('float32')
    query_embs = query_embeddings.astype('float32')

    if torch.cuda.is_available():
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, corpus_embs.shape[1])
    else:
        index = faiss.IndexFlatIP(corpus_embs.shape[1])
    index.add(corpus_embs)

    # 🔥 IMPROVED: Search more candidates for better filtering
    search_k = min(K * 30, len(corpus_texts))
    print(f"\n🔍 Searching top-{search_k} candidates per query")

    D, I = index.search(query_embs, search_k)

    result = defaultdict(list)
    stats = {'full_k': 0, 'partial_k': 0, 'no_candidates': 0}
    query_details = []

    print("\n🎯 Smart negative filtering with optimal similarity ranges:")
    for i, (neighbors, scores) in enumerate(
            tqdm(zip(I, D), total=len(queries), desc="Mining hard negatives", ncols=100)):
        q = queries[i]
        positives = pos_dict[q]
        valid_negatives = []

        # 🔥 IMPROVED: Use optimal similarity ranges for better discrimination
        candidate_pool = []

        for idx, score in zip(neighbors, scores):
            candidate = corpus_texts[idx]

            # Skip invalid candidates
            if (not candidate or candidate == q or candidate in positives):
                continue

            # 🔥 CRITICAL: Smart similarity filtering
            # We want challenging but clearly negative examples
            # Optimal range: 0.15-0.45 (challenging but discriminable)
            if 0.15 <= score <= 0.45:
                candidate_pool.append((candidate, score, 'optimal'))
            # Include some moderately similar for diversity
            elif 0.45 < score <= 0.6 and len(candidate_pool) < K // 2:
                candidate_pool.append((candidate, score, 'moderate'))
            # Include some lower similarity for easy negatives
            elif 0.05 <= score < 0.15 and len(candidate_pool) < K // 3:
                candidate_pool.append((candidate, score, 'easy'))

        # Sort by similarity (higher first for challenging negatives)
        candidate_pool.sort(key=lambda x: x[1], reverse=True)

        # 🔥 STRATEGIC SELECTION: Mix of challenging and easy negatives
        optimal_negs = [c for c in candidate_pool if c[2] == 'optimal']
        moderate_negs = [c for c in candidate_pool if c[2] == 'moderate']
        easy_negs = [c for c in candidate_pool if c[2] == 'easy']

        # Priority: optimal > moderate > easy
        selected = []
        selected.extend(optimal_negs[:min(len(optimal_negs), K)])
        remaining = K - len(selected)
        if remaining > 0 and moderate_negs:
            selected.extend(moderate_negs[:min(len(moderate_negs), remaining)])
        remaining = K - len(selected)
        if remaining > 0 and easy_negs:
            selected.extend(easy_negs[:min(len(easy_negs), remaining)])

        valid_negatives = [c[0] for c in selected[:K]]

        # 🔥 FALLBACK STRATEGY: Broader search if needed
        if len(valid_negatives) < K:
            additional_candidates = []
            for idx, score in zip(neighbors, scores):
                if len(valid_negatives) + len(additional_candidates) >= K:
                    break
                candidate = corpus_texts[idx]
                if (candidate and candidate != q and candidate not in positives and
                        candidate not in valid_negatives and candidate not in additional_candidates):
                    if 0.02 <= score <= 0.7:  # Much broader range
                        additional_candidates.append(candidate)
            valid_negatives.extend(additional_candidates[:K - len(valid_negatives)])

        # Store results
        query_info = {
            'query': q,
            'negatives_found': len(valid_negatives),
            'similarity_range': f"{min([s for _, s, _ in candidate_pool]):.3f}-{max([s for _, s, _ in candidate_pool]):.3f}" if candidate_pool else "N/A",
            'types_selected': {typ: len([c for c in selected if c[2] == typ]) for typ in
                               ['optimal', 'moderate', 'easy']}
        }
        query_details.append(query_info)

        # Debug output for first queries
        if i < 5:
            sim_scores = [s for _, s, _ in candidate_pool[:5]]
            avg_sim = np.mean(sim_scores) if sim_scores else 0
            print(f"   Query {i}: {len(valid_negatives)}/{K} negatives "
                  f"(avg similarity: {avg_sim:.3f}, types: {query_info['types_selected']})")

        # Update stats
        if len(valid_negatives) == K:
            stats['full_k'] += 1
        elif len(valid_negatives) > 0:
            stats['partial_k'] += 1
        else:
            stats['no_candidates'] += 1

        result[q] = valid_negatives[:K]

    # Clean up FAISS
    if torch.cuda.is_available():
        del index

    # 🔥 ENHANCED QUALITY ANALYSIS
    print(f"\n" + "=" * 60)
    print("🎯 HARD NEGATIVE QUALITY ANALYSIS")
    print("=" * 60)

    # Calculate average similarity of selected negatives
    all_similarities = []
    type_distribution = defaultdict(int)

    for detail in query_details:
        type_distribution['optimal'] += detail['types_selected']['optimal']
        type_distribution['moderate'] += detail['types_selected']['moderate']
        type_distribution['easy'] += detail['types_selected']['easy']

    total_negatives = sum(type_distribution.values())

    print(f"📊 Negative Type Distribution:")
    print(
        f"   • Optimal (0.15-0.45): {type_distribution['optimal']} ({type_distribution['optimal'] / total_negatives * 100:.1f}%)")
    print(
        f"   • Moderate (0.45-0.60): {type_distribution['moderate']} ({type_distribution['moderate'] / total_negatives * 100:.1f}%)")
    print(
        f"   • Easy (0.05-0.15): {type_distribution['easy']} ({type_distribution['easy'] / total_negatives * 100:.1f}%)")

    # Quality assessment
    optimal_percentage = type_distribution['optimal'] / total_negatives * 100
    if optimal_percentage > 60:
        quality_rating = "🎉 EXCELLENT"
    elif optimal_percentage > 40:
        quality_rating = "✅ GOOD"
    elif optimal_percentage > 20:
        quality_rating = "⚠️  FAIR"
    else:
        quality_rating = "🚨 POOR"

    print(f"📈 Quality Assessment: {quality_rating}")
    print(f"   • Optimal challenging negatives: {optimal_percentage:.1f}%")

    # Overall statistics
    print(f"\n📈 Overall Statistics:")
    print(f"   • Total queries processed: {len(queries)}")
    print(f"   • Queries with full {K} negatives: {stats['full_k']} ({stats['full_k'] / len(queries) * 100:.1f}%)")
    print(f"   • Queries with partial negatives: {stats['partial_k']} ({stats['partial_k'] / len(queries) * 100:.1f}%)")
    print(
        f"   • Queries with NO candidates: {stats['no_candidates']} ({stats['no_candidates'] / len(queries) * 100:.1f}%)")

    total_negatives_generated = sum(len(negs) for negs in result.values())
    coverage_rate = (total_negatives_generated / (len(queries) * K)) * 100

    print(f"\n🎯 Success Metrics:")
    print(f"   • Total negatives generated: {total_negatives_generated}/{len(queries) * K}")
    print(f"   • Coverage rate: {coverage_rate:.1f}%")
    print(f"   • Average negatives per query: {total_negatives_generated / len(queries):.1f}")

    write_pickle(result, output_path)
    print(f"\n💾 High-quality hard negatives saved to: {output_path}")
    print("=" * 60)

    return result


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

    print("🚀 Starting improved hard negative generation...")
    print(f"📝 Configuration: K={config['K']}, Model={config['model_name']}")
    print(f"🎯 Strategy: Quality over quantity with optimal similarity ranges")

    for split in config['sets']:
        print(f"\n" + "=" * 50)
        print(f"Processing '{split}' split")
        print("=" * 50)
        process_dataset(split, model)


if __name__ == '__main__':
    main()