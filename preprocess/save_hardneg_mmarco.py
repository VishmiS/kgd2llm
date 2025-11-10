# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/preprocess/save_hardneg_mmarco.py

import sys
import pathlib

project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import os
import csv
import torch
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

# OPTIMIZED config for speed
config = {
    'sets': ['train', 'val'],
    'base_dir': 'dataset/ms_marco',
    'output_base': 'outputs/neg_mmarco',
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'K': 8,
    'max_seq_len': 128,  # Reduced from 250
    'use_sample': False,
    'sample_limit': 10000,
    'batch_size': 512,  # Increased batch size
    'query_batch_size': 50000,  # Larger FAISS batches
    'use_gpu': True,  # Use GPU for encoding (much faster)
    'faiss_gpu': True,  # Use GPU for FAISS
    'cache_dir': '/root/pycharm_semanticsearch/cache',
    'num_threads': 16,  # Use more CPU threads
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


class FastBertModel:
    def __init__(self, model_name_or_path, max_seq_length=128, encoder_type='MEAN'):
        self.encoder_type = EncoderType.from_string(encoder_type)
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.plm_model = AutoModel.from_pretrained(model_name_or_path)

        # Use GPU if available and configured
        if config['use_gpu'] and torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device("cpu")
            print("⚠️ Using CPU")

        self.plm_model.to(self.device)

        # Enable mixed precision for 2-3x speedup
        self.use_amp = True if self.device.type == 'cuda' else False
        if self.use_amp:
            print("🔸 Mixed precision enabled")

        # Compile model for PyTorch 2.0+ (30% speedup)
        if hasattr(torch, 'compile'):
            self.plm_model = torch.compile(self.plm_model, mode="reduce-overhead")
            print("🔸 Model compilation enabled")

    def get_sentence_embeddings(self, input_ids, attention_mask):
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            output = self.plm_model(input_ids=input_ids, attention_mask=attention_mask)

            if self.encoder_type == EncoderType.CLS:
                embeddings = output.last_hidden_state[:, 0]
            elif self.encoder_type == EncoderType.MEAN:
                token_embeddings = output.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                embeddings = sum_embeddings / sum_mask

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            return embeddings

    def batch_to_device(self, batch):
        return {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

    def encode_fast(self, sentences: List[str], batch_size=512, max_seq_length=None):
        """Optimized encoding with parallel processing"""
        self.plm_model.eval()
        max_seq_length = max_seq_length or self.max_seq_length

        # Pre-allocate results array
        all_embeddings = np.zeros((len(sentences), 384), dtype=np.float32)  # MiniLM has 384 dim

        with torch.no_grad():
            for start in trange(0, len(sentences), batch_size,
                                desc="🚀 Fast Encoding", dynamic_ncols=True):
                end = start + batch_size
                batch = sentences[start:end]

                # Tokenize without slow python loops
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors='pt'
                )
                inputs = self.batch_to_device(inputs)

                embs = self.get_sentence_embeddings(inputs["input_ids"], inputs["attention_mask"])

                # Direct numpy conversion to avoid CPU overhead
                all_embeddings[start:end] = embs.float().cpu().numpy()

        return all_embeddings


def save_corpus_embeddings_fast(model, tsv_path, output_dir, split):
    """Fast corpus processing with memory mapping"""
    ensure_dir(output_dir)
    ensure_dir(config['cache_dir'])

    cache_path = os.path.join(config['cache_dir'], f'{split}_corpus_embeddings.npy')
    corpus_texts_path = os.path.join(output_dir, 'corpus_texts.pkl')

    # FAST corpus reading
    print(f"📖 Fast reading corpus from {tsv_path}...")
    corpus_set = set()

    # Use file size for better progress estimation
    file_size = os.path.getsize(tsv_path)

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc="Reading corpus", total=file_size // 1000):  # rough estimate
            text = row['text'].strip()
            if text:
                corpus_set.add(text)
            if config.get('use_sample', False) and len(corpus_set) >= config.get('sample_limit', float('inf')):
                break

    corpus = list(corpus_set)
    print(f"📚 Corpus size: {len(corpus):,}")

    # Save corpus texts
    write_pickle(corpus, corpus_texts_path)

    # Fast encoding with cache
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path, mmap_mode='r')
        if embeddings.shape[0] == len(corpus):
            print(f"✅ Loaded cached corpus embeddings from {cache_path}")
        else:
            print(f"⚠️ Cache size mismatch - re-encoding corpus...")
            embeddings = model.encode_fast(corpus, batch_size=config['batch_size'])
            np.save(cache_path, embeddings)
    else:
        print(f"⚙️ Fast encoding corpus...")
        embeddings = model.encode_fast(corpus, batch_size=config['batch_size'])
        np.save(cache_path, embeddings)

    embeddings_path = os.path.join(output_dir, 'corpus_embeddings.pkl')
    write_pickle(embeddings, embeddings_path)

    return corpus_texts_path, embeddings_path


def save_queries_and_hard_negatives_fast(model, positive_path, corpus_texts_path,
                                         corpus_embeddings_path, output_path, split):
    """Ultra-fast hard negative generation"""

    # FAST query reading
    print(f"📖 Fast reading queries from {positive_path}...")
    queries = []
    pos_dict = defaultdict(set)

    with open(positive_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc="Reading queries"):
            q = row.get('sentence1', '').strip()
            a = row.get('sentence2', '').strip()
            if q and a:
                queries.append(q)
                pos_dict[q].add(a)
                if config.get('use_sample', False) and len(queries) >= config.get('sample_limit', float('inf')):
                    break

    # Remove duplicates while preserving order
    seen_queries = set()
    unique_queries = []
    for q in queries:
        if q not in seen_queries:
            seen_queries.add(q)
            unique_queries.append(q)
    queries = unique_queries

    print(f"✅ Loaded {len(queries):,} unique queries")

    # Load corpus
    corpus_texts = load_pickle(corpus_texts_path)
    corpus_embeddings = load_pickle(corpus_embeddings_path)
    print(f"✅ Loaded corpus: {len(corpus_texts):,} texts")

    # Fast query encoding with cache
    query_cache_path = os.path.join(config['cache_dir'], f'{split}_query_embeddings.npy')
    if os.path.exists(query_cache_path):
        query_embeddings = np.load(query_cache_path)
        print(f"✅ Loaded cached query embeddings")

        # CRITICAL FIX: Ensure query embeddings match queries count
        if query_embeddings.shape[0] != len(queries):
            print(f"⚠️ Cache mismatch: {query_embeddings.shape[0]:,} embeddings vs {len(queries):,} queries")
            print(f"🔄 Re-encoding queries to fix mismatch...")
            query_embeddings = model.encode_fast(queries, batch_size=config['batch_size'], max_seq_length=64)
            np.save(query_cache_path, query_embeddings)
    else:
        print(f"⚙️ Fast encoding queries...")
        query_embeddings = model.encode_fast(queries, batch_size=config['batch_size'], max_seq_length=64)
        np.save(query_cache_path, query_embeddings)

    # Convert to float32 for FAISS
    corpus_embs = corpus_embeddings.astype('float32')
    query_embs = query_embeddings.astype('float32')

    # CRITICAL FIX: Final validation before FAISS
    if query_embs.shape[0] != len(queries):
        print(f"❌ CRITICAL ERROR: Query embeddings ({query_embs.shape[0]:,}) don't match queries ({len(queries):,})")
        print(f"🔄 Trimming queries to match embeddings...")
        # Keep only the queries that have corresponding embeddings
        queries = queries[:query_embs.shape[0]]
        print(f"✅ Adjusted to {len(queries):,} queries")

    # ULTRA-FAST FAISS on GPU
    print("🔨 Building FAISS index...")

    if config['faiss_gpu'] and torch.cuda.is_available():
        # Use GPU FAISS for 10-50x speedup
        res = faiss.StandardGpuResources()
        index = faiss.IndexFlatIP(corpus_embs.shape[1])
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
        gpu_index.add(corpus_embs)
        print(f"✅ GPU FAISS index built with {gpu_index.ntotal:,} vectors")
    else:
        # Fallback to CPU
        index = faiss.IndexFlatIP(corpus_embs.shape[1])
        index.add(corpus_embs)
        gpu_index = index
        print(f"✅ CPU FAISS index built with {index.ntotal:,} vectors")

    # FAST batch search
    search_k = min(config['K'] + 50, len(corpus_texts))  # Don't search more than corpus size
    print(f"🔍 Fast searching for {search_k} neighbors...")

    # Single batch search if possible (much faster)
    if query_embs.shape[0] <= 100000:  # Fit in GPU memory
        D, I = gpu_index.search(query_embs, search_k)
    else:
        # Batched search for very large query sets
        all_scores, all_indices = [], []
        batch_size = min(config['query_batch_size'], query_embs.shape[0])

        for start in trange(0, query_embs.shape[0], batch_size,
                            desc="FAISS Search", dynamic_ncols=True):
            end = min(start + batch_size, query_embs.shape[0])
            D_batch, I_batch = gpu_index.search(query_embs[start:end], search_k)
            all_scores.append(D_batch)
            all_indices.append(I_batch)

        D = np.vstack(all_scores)
        I = np.vstack(all_indices)

    # CRITICAL FIX: Validate FAISS results before processing
    if I.shape[0] != len(queries):
        print(f"⚠️ FAISS returned {I.shape[0]:,} results for {len(queries):,} queries")
        min_size = min(I.shape[0], len(queries))
        I = I[:min_size]
        D = D[:min_size]
        queries = queries[:min_size]
        print(f"✅ Adjusted to {len(queries):,} queries for processing")

    print("🎯 Fast filtering hard negatives...")

    # OPTIMIZED filtering using numpy operations
    result = {}
    corpus_texts_arr = np.array(corpus_texts, dtype=object)

    for i in trange(len(queries), desc="Generating negatives"):
        q = queries[i]
        neighbors = I[i]
        scores = D[i]

        # Vectorized filtering
        candidates = corpus_texts_arr[neighbors]

        # Create mask for valid negatives
        valid_mask = np.ones(len(candidates), dtype=bool)

        # Filter out empty, query itself, and positives
        for j, candidate in enumerate(candidates):
            if (not candidate or
                    candidate == q or
                    candidate in pos_dict[q] or
                    scores[j] < 0.1):
                valid_mask[j] = False

        valid_candidates = candidates[valid_mask]
        valid_candidates = valid_candidates[:config['K']]  # Take top K

        result[q] = valid_candidates.tolist()

    # Final validation
    if len(result) != len(queries):
        print(f"❌ Processing mismatch: {len(result):,} results vs {len(queries):,} queries")
        # Add any missing queries with empty negatives
        processed_queries = set(result.keys())
        for i, q in enumerate(queries):
            if q not in processed_queries:
                result[q] = []
        print(f"✅ Added empty negatives for {len(queries) - len(processed_queries):,} missing queries")

    # Save results
    write_pickle(result, output_path)

    # Statistics
    neg_counts = [len(negs) for negs in result.values()]
    print(f"✅ Saved {len(result):,} queries → {output_path}")
    print(f"📊 Hard negatives stats: avg={np.mean(neg_counts):.2f}, "
          f"min={np.min(neg_counts)}, max={np.max(neg_counts)}")

    return result


def process_dataset_fast(split: str, model: FastBertModel):
    """Fast processing pipeline"""
    base_dir = config['base_dir']
    input_dir = os.path.join(base_dir, split)
    output_dir = os.path.join(config['output_base'], split)

    ensure_dir(input_dir)
    ensure_dir(output_dir)

    corpus_path = os.path.join(input_dir, 'corpus.tsv')
    positive_path = os.path.join(input_dir, 'positives.tsv')
    output_path = os.path.join(output_dir, 'query_hard_negatives.pkl')

    # Validate files
    for f in [corpus_path, positive_path]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Missing file: {f}")

    print(f"\n🚀 ULTRA-FAST Processing {split} split...")

    # Set threads for numpy/faiss
    if config.get('num_threads'):
        faiss.omp_set_num_threads(config['num_threads'])
        print(f"🔸 Using {config['num_threads']} CPU threads")

    # Step 1: Fast corpus processing
    print("📦 Step 1: Fast corpus processing...")
    corpus_texts_path, corpus_embeddings_path = save_corpus_embeddings_fast(
        model, corpus_path, output_dir, split
    )

    # Step 2: Fast hard negative generation
    print("📦 Step 2: Fast hard negative generation...")
    save_queries_and_hard_negatives_fast(
        model=model,
        positive_path=positive_path,
        corpus_texts_path=corpus_texts_path,
        corpus_embeddings_path=corpus_embeddings_path,
        output_path=output_path,
        split=split
    )

    print(f"✅ Completed {split} split!")


def main():
    """Main function with performance optimizations"""
    set_seed(42)

    # Create directories
    ensure_dir(config['output_base'])
    ensure_dir(config['cache_dir'])

    print("🤖 Initializing FAST model...")
    model = FastBertModel(
        config['model_name'],
        config['max_seq_len'],
        encoder_type='MEAN'
    )

    print("🔧 OPTIMIZED Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Process splits
    for split in config['sets']:
        print(f"\n{'=' * 60}")
        print(f"🚀 ULTRA-FAST Processing {split.upper()} split")
        print(f"{'=' * 60}")

        try:
            process_dataset_fast(split, model)
        except Exception as e:
            print(f"❌ Error processing {split}: {e}")
            import traceback
            traceback.print_exc()

            # Fallback to CPU if GPU fails
            if "CUDA" in str(e):
                print("🔄 Falling back to CPU mode...")
                config['use_gpu'] = False
                config['faiss_gpu'] = False
                model = FastBertModel(
                    config['model_name'],
                    config['max_seq_len'],
                    encoder_type='MEAN'
                )
                process_dataset_fast(split, model)
            continue

    print(f"\n🎉 All splits processed at HIGH SPEED!")
    print(f"📁 Output: {config['output_base']}")
    print(f"💾 Cache: {config['cache_dir']}")


if __name__ == '__main__':
    main()