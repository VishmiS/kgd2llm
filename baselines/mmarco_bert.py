# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/baselines/mmarco_bert.py

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm
import faiss
import time
import os
import pickle
import hashlib


class BERTEvaluator:
    """Base class for BERT evaluation."""

    def __init__(self, model_name='bert-base-uncased', batch_size=32, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _encode_texts(self, texts):
        embeddings = []
        total_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            encoded = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            # Mean pooling over token embeddings
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())

            # Print progress every 10 batches or at the end
            batch_num = (i // self.batch_size) + 1
            if batch_num % 10 == 0 or batch_num == total_batches:
                print(f"Encoded {min(i + self.batch_size, len(texts))}/{len(texts)} documents...")

        return np.vstack(embeddings)

    def _calculate_mrr(self, retrieved_ids, relevant_ids):
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def _calculate_recall(self, retrieved_ids, relevant_ids):
        hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
        return hits / len(relevant_ids) if relevant_ids else 0.0


class BERTEvaluatorFAISS(BERTEvaluator):
    """BERT evaluation using FAISS for fast retrieval."""

    def __init__(self, model_name='bert-base-uncased', batch_size=32, device=None, cache_dir='./embedding_cache'):
        super().__init__(model_name, batch_size, device)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, corpus, model_name):
        """Generate a unique cache key based on corpus content and model name."""
        corpus_content = ''.join(f"{k}:{v}" for k, v in sorted(corpus.items()))
        key_string = f"{model_name}_{corpus_content}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_cached_embeddings(self, cache_key):
        """Load cached embeddings and FAISS index if they exist."""
        embeddings_file = os.path.join(self.cache_dir, f"{cache_key}_embeddings.pkl")
        doc_ids_file = os.path.join(self.cache_dir, f"{cache_key}_doc_ids.pkl")
        faiss_index_file = os.path.join(self.cache_dir, f"{cache_key}_faiss.index")

        if os.path.exists(embeddings_file) and os.path.exists(doc_ids_file) and os.path.exists(faiss_index_file):
            print("Loading cached embeddings...")
            with open(embeddings_file, 'rb') as f:
                doc_embeddings = pickle.load(f)
            with open(doc_ids_file, 'rb') as f:
                doc_ids = pickle.load(f)

            # Load FAISS index
            index = faiss.read_index(faiss_index_file)
            return doc_embeddings, doc_ids, index
        return None, None, None

    def _save_embeddings(self, cache_key, doc_embeddings, doc_ids, index):
        """Save embeddings and FAISS index to cache."""
        embeddings_file = os.path.join(self.cache_dir, f"{cache_key}_embeddings.pkl")
        doc_ids_file = os.path.join(self.cache_dir, f"{cache_key}_doc_ids.pkl")
        faiss_index_file = os.path.join(self.cache_dir, f"{cache_key}_faiss.index")

        with open(embeddings_file, 'wb') as f:
            pickle.dump(doc_embeddings, f)
        with open(doc_ids_file, 'wb') as f:
            pickle.dump(doc_ids, f)
        faiss.write_index(index, faiss_index_file)
        print(f"Embeddings cached with key: {cache_key}")

    def get_document_embeddings(self, corpus):
        """Get document embeddings, using cache if available."""
        cache_key = self._get_cache_key(corpus, self.model_name)

        # Try to load from cache
        doc_embeddings, doc_ids, index = self._load_cached_embeddings(cache_key)

        if doc_embeddings is not None:
            print("Using cached document embeddings")
            return doc_embeddings, doc_ids, index

        # Generate new embeddings
        print("Encoding documents...")
        doc_ids = list(corpus.keys())
        doc_texts = list(corpus.values())

        print(f"Starting to encode {len(doc_texts)} documents...")
        doc_embeddings = self._encode_texts(doc_texts)
        print("Document encoding completed!")

        # Create FAISS index
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)

        # Normalize embeddings for cosine similarity and add to index
        faiss.normalize_L2(doc_embeddings)
        index.add(doc_embeddings)

        # Save to cache
        self._save_embeddings(cache_key, doc_embeddings, doc_ids, index)

        return doc_embeddings, doc_ids, index

    def evaluate(self, corpus, queries, qrels, k=10, use_cache=True):
        if use_cache:
            doc_embeddings, self.doc_ids, index = self.get_document_embeddings(corpus)
        else:
            self.doc_ids = list(corpus.keys())
            doc_texts = list(corpus.values())
            print("Encoding documents...")
            doc_embeddings = self._encode_texts(doc_texts)

            # Create FAISS index
            dimension = doc_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(doc_embeddings)
            index.add(doc_embeddings)

        mrr_scores, recall_scores = [], []

        print("Evaluating queries...")
        for query_id, query_text in tqdm(queries.items()):
            relevant_docs = set(qrels.get(query_id, {}).keys())
            if not relevant_docs:
                continue

            query_embedding = self._encode_texts([query_text])
            faiss.normalize_L2(query_embedding)

            similarities, indices = index.search(query_embedding, k)
            top_k_doc_ids = [self.doc_ids[idx] for idx in indices[0]]

            mrr_scores.append(self._calculate_mrr(top_k_doc_ids, relevant_docs))
            recall_scores.append(self._calculate_recall(top_k_doc_ids, relevant_docs))

        return {
            f"MRR@{k}": np.mean(mrr_scores),
            f"Recall@{k}": np.mean(recall_scores),
            "num_queries": len(mrr_scores)
        }


def load_tsv_data(corpus_file, queries_file, qrels_file):
    corpus, queries, qrels = {}, {}, defaultdict(dict)

    # Load corpus
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                corpus[parts[0]] = '\t'.join(parts[1:])
            else:
                print(f"Warning: Skipping malformed corpus line {line_num}: {line.strip()}")

    # Load queries
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = '\t'.join(parts[1:])
            else:
                print(f"Warning: Skipping malformed query line {line_num}: {line.strip()}")

    # Load qrels
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip() or line.lower().startswith("query"):
                continue  # skip empty lines or header
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    relevance = int(parts[2])
                    qrels[parts[0]][parts[1]] = relevance
                except ValueError:
                    print(f"Warning: Skipping line {line_num} in qrels with invalid relevance: {line.strip()}")
            else:
                print(f"Warning: Skipping malformed qrels line {line_num}: {line.strip()}")

    return corpus, queries, qrels


class EmbeddingManager:
    """Manager class to handle embedding caching and retrieval for multiple models."""

    def __init__(self, cache_dir='./embedding_cache'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_embeddings_for_model(self, model_name, corpus, batch_size=32, device=None):
        """Get embeddings for a specific model, using cache if available."""
        evaluator = BERTEvaluatorFAISS(
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            cache_dir=self.cache_dir
        )
        doc_embeddings, doc_ids, index = evaluator.get_document_embeddings(corpus)
        return doc_embeddings, doc_ids, index, evaluator

    def list_cached_embeddings(self):
        """List all cached embedding files."""
        cache_files = []
        for file in os.listdir(self.cache_dir):
            if file.endswith('_embeddings.pkl'):
                cache_key = file.replace('_embeddings.pkl', '')
                cache_files.append(cache_key)
        return cache_files


def main():
    # Update these paths
    corpus_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/corpus.tsv"
    queries_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/queries.tsv"
    qrels_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/qrels.tsv"

    corpus, queries, qrels = load_tsv_data(corpus_file, queries_file, qrels_file)

    # Example 1: Standard evaluation with caching
    print("=== BERT Evaluation with Caching ===")
    evaluator = BERTEvaluatorFAISS(batch_size=32, device='cuda')
    start_time = time.time()
    results = evaluator.evaluate(corpus, queries, qrels, k=10, use_cache=True)
    end_time = time.time()

    print("\n" + "=" * 60)
    print("BERT EVALUATION RESULTS")
    print("=" * 60)
    for metric, score in results.items():
        if metric != "num_queries":
            print(f"{metric}: {score:.4f}")
        else:
            print(f"{metric}: {score}")
    print(f"Evaluation time: {(end_time - start_time) / 60:.2f} minutes")
    print("=" * 60)

    # Example 2: Using EmbeddingManager for multiple models
    print("\n=== Using Embedding Manager for Multiple Models ===")
    manager = EmbeddingManager()

    # Get embeddings for different models
    models = ['bert-base-uncased']

    for model_name in models:
        print(f"\nLoading embeddings for: {model_name}")
        try:
            doc_embeddings, doc_ids, index, evaluator = manager.get_embeddings_for_model(
                model_name, corpus, batch_size=32, device='cuda'
            )
            print(f"Successfully loaded embeddings for {model_name}")
            print(f"Embedding shape: {doc_embeddings.shape}")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")

    # List cached embeddings
    print(f"\nCached embeddings: {manager.list_cached_embeddings()}")


if __name__ == "__main__":
    main()