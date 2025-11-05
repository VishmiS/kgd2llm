# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/baselines/mmarco_biobert.py

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import json
import time
from collections import defaultdict
import faiss
import os
import pickle
import hashlib
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioBERTEvaluatorWithCache:
    def __init__(self, corpus, queries, qrels, model_name='dmis-lab/biobert-v1.1',
                 cache_dir='./embedding_cache_biobert'):
        """
        Initialize BioBERT evaluator with caching

        Args:
            corpus: dict with doc_id as key and document text as value
            queries: dict with query_id as key and query text as value
            qrels: dict with query_id as key and dict of relevant doc_ids as value
            model_name: pre-trained BioBERT model name
            cache_dir: directory to cache embeddings
        """
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.cache_dir = cache_dir
        self.model_name = model_name

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Load BioBERT model and tokenizer
        logger.info(f"Loading BioBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Preprocess and encode corpus with caching
        self.doc_ids = list(corpus.keys())
        self.doc_texts = list(corpus.values())

        logger.info("Loading/encoding corpus...")
        self.doc_embeddings, self.index = self._get_cached_embeddings()

    def _get_cache_key(self):
        """Generate a unique cache key based on corpus content and model name."""
        corpus_content = ''.join(f"{k}:{v}" for k, v in sorted(self.corpus.items()))
        key_string = f"{self.model_name}_{corpus_content}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _load_cached_embeddings(self, cache_key):
        """Load cached embeddings and FAISS index if they exist."""
        embeddings_file = os.path.join(self.cache_dir, f"{cache_key}_embeddings.pkl")
        doc_ids_file = os.path.join(self.cache_dir, f"{cache_key}_doc_ids.pkl")
        faiss_index_file = os.path.join(self.cache_dir, f"{cache_key}_faiss.index")

        if os.path.exists(embeddings_file) and os.path.exists(doc_ids_file) and os.path.exists(faiss_index_file):
            logger.info("Loading cached embeddings...")
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
        logger.info(f"Embeddings cached with key: {cache_key}")

    def _encode_corpus(self, texts, batch_size=32):
        """Encode all documents in the corpus"""
        all_embeddings = []

        pbar = tqdm(total=len(texts), desc="Encoding documents", unit="doc")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Encode
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling for document representation
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(embeddings)

            pbar.update(len(batch_texts))

        pbar.close()

        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)

        # Normalize embeddings for cosine similarity
        all_embeddings = normalize(all_embeddings, axis=1, norm='l2')

        return all_embeddings

    def _get_cached_embeddings(self):
        """Get document embeddings, using cache if available."""
        cache_key = self._get_cache_key()

        # Try to load from cache
        doc_embeddings, doc_ids, index = self._load_cached_embeddings(cache_key)

        if doc_embeddings is not None:
            logger.info("Using cached document embeddings")
            # Verify doc_ids match
            if doc_ids == self.doc_ids:
                return doc_embeddings, index
            else:
                logger.info("Document IDs don't match, regenerating embeddings...")

        # Generate new embeddings
        logger.info("Encoding documents...")
        doc_embeddings = self._encode_corpus(self.doc_texts)
        logger.info("Document encoding completed!")

        # Create FAISS index
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(doc_embeddings.astype(np.float32))

        # Save to cache
        self._save_embeddings(cache_key, doc_embeddings, self.doc_ids, index)

        return doc_embeddings, index

    def _encode_query(self, query_text):
        """Encode a single query"""
        inputs = self.tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        # Normalize for cosine similarity
        query_embedding = normalize(query_embedding, axis=1, norm='l2')

        return query_embedding.astype(np.float32)

    def _encode_queries_batch(self, query_texts, batch_size=32):
        """Encode multiple queries in batches for better performance"""
        all_embeddings = []

        for i in range(0, len(query_texts), batch_size):
            batch_texts = query_texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(batch_embeddings)

        # Concatenate and normalize
        all_embeddings = np.vstack(all_embeddings)
        all_embeddings = normalize(all_embeddings, axis=1, norm='l2')

        return all_embeddings.astype(np.float32)

    def search(self, query_text, k=10):
        """Search for similar documents using BioBERT"""
        query_embedding = self._encode_query(query_text)

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, k)

        # Convert to document IDs and scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            doc_id = self.doc_ids[idx]
            results.append((doc_id, float(score)))

        return results

    def evaluate(self, k=10, relevance_threshold=1.0, query_batch_size=32):
        """
        Evaluate BioBERT on MRR@k and Recall@k with optimized query processing

        Args:
            k: cutoff for evaluation metrics
            relevance_threshold: minimum relevance score to consider a document relevant
            query_batch_size: batch size for query encoding

        Returns:
            dict with MRR@k and Recall@k scores
        """
        mrr_scores = []
        recall_scores = []

        # Filter queries that have relevant documents
        valid_queries = {}
        for query_id, query_text in self.queries.items():
            relevant_docs_dict = self.qrels.get(query_id, {})
            relevant_docs = set(doc_id for doc_id, rel_score in relevant_docs_dict.items()
                                if rel_score >= relevance_threshold)
            if relevant_docs:
                valid_queries[query_id] = (query_text, relevant_docs)

        logger.info(f"Evaluating {len(valid_queries)} queries with relevant documents...")

        if not valid_queries:
            logger.error("No queries with relevant documents found!")
            return {
                f"MRR@{k}": 0.0,
                f"Recall@{k}": 0.0,
                "num_queries": 0,
                "relevance_threshold": relevance_threshold
            }

        # Encode all valid queries in batches for better performance
        query_ids = list(valid_queries.keys())
        query_texts = [valid_queries[qid][0] for qid in query_ids]

        logger.info("Encoding queries in batches...")
        query_embeddings = self._encode_queries_batch(query_texts, batch_size=query_batch_size)

        # Search for all queries at once
        logger.info("Performing similarity search...")
        scores, indices = self.index.search(query_embeddings, k)

        # Calculate metrics
        for i, query_id in enumerate(tqdm(query_ids, desc="Evaluating queries")):
            query_text, relevant_docs = valid_queries[query_id]

            top_k_doc_ids = [self.doc_ids[idx] for idx in indices[i]]

            # Calculate MRR@k
            mrr = self._calculate_mrr(top_k_doc_ids, relevant_docs)
            mrr_scores.append(mrr)

            # Calculate Recall@k
            recall = self._calculate_recall(top_k_doc_ids, relevant_docs)
            recall_scores.append(recall)

        # Calculate final metrics
        if mrr_scores:
            final_mrr = np.mean(mrr_scores)
            final_recall = np.mean(recall_scores)
        else:
            final_mrr = 0.0
            final_recall = 0.0
            logger.warning("No queries with relevant documents found!")

        return {
            f"MRR@{k}": final_mrr,
            f"Recall@{k}": final_recall,
            "num_queries": len(mrr_scores),
            "relevance_threshold": relevance_threshold
        }

    def _calculate_mrr(self, retrieved_docs, relevant_docs):
        """Calculate MRR for a single query"""
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / rank
        return 0.0

    def _calculate_recall(self, retrieved_docs, relevant_docs):
        """Calculate Recall for a single query"""
        if not relevant_docs:
            return 0.0

        retrieved_relevant = len([doc for doc in retrieved_docs if doc in relevant_docs])
        return retrieved_relevant / len(relevant_docs)


def load_tsv_data(corpus_file, queries_file, qrels_file):
    """Load TSV files for corpus, queries, and qrels with header handling"""
    corpus, queries, qrels = {}, {}, defaultdict(dict)

    # Load corpus - skip header
    logger.info(f"Loading corpus from {corpus_file}")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            # Skip header line
            if i == 0 and parts[0].lower() in ['corpus_id', 'doc_id', 'document_id', 'id']:
                logger.info("Skipping corpus header row")
                continue
            if len(parts) >= 2:
                corpus[parts[0]] = '\t'.join(parts[1:])
            if i < 3:  # Print first few lines for debugging
                logger.info(f"Corpus line {i}: {parts[0]} -> {parts[1][:50]}...")

    # Load queries - skip header
    logger.info(f"Loading queries from {queries_file}")
    with open(queries_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            # Skip header line
            if i == 0 and parts[0].lower() in ['query_id', 'qid', 'id', 'query']:
                logger.info("Skipping queries header row")
                continue
            if len(parts) >= 2:
                queries[parts[0]] = '\t'.join(parts[1:])
            if i < 3:  # Print first few lines for debugging
                logger.info(f"Query line {i}: {parts[0]} -> {parts[1][:50]}...")

    # Load qrels - skip header and handle float relevance scores
    logger.info(f"Loading qrels from {qrels_file}")
    qrels_count = 0
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            # Skip header line
            if i == 0 and any(header in line.lower() for header in
                              ['query_id', 'qid', 'corpus_id', 'doc_id', 'rel', 'relevance']):
                logger.info("Skipping qrels header row")
                continue

            # Try tab separator first (most likely for TSV)
            parts = line.strip().split('\t')
            if len(parts) < 3:
                # Try space separator
                parts = line.strip().split()

            if len(parts) >= 3:
                try:
                    query_id, doc_id, relevance_str = parts[0], parts[1], parts[2]
                    # Handle both integer and float relevance scores
                    relevance = float(relevance_str)
                    # Consider documents with relevance > 0 as relevant
                    if relevance > 0:
                        qrels[query_id][doc_id] = relevance
                        qrels_count += 1
                    if i < 5:  # Print first few lines for debugging
                        logger.info(f"Qrels line {i}: {query_id} -> {doc_id} (rel: {relevance})")
                except (ValueError, IndexError) as e:
                    logger.debug(f"Could not parse qrels line {i}: {line.strip()} - {e}")
                    continue

    logger.info(
        f"Loaded: {len(corpus)} documents, {len(queries)} queries, {qrels_count} relevance judgments (relevance > 0)")

    # Debug: show query IDs and which have relevance judgments
    queries_with_judgments = [qid for qid in queries if qid in qrels and qrels[qid]]
    logger.info(f"Queries with relevance judgments: {len(queries_with_judgments)}")

    # Show some statistics about relevance judgments per query
    for qid in queries_with_judgments[:5]:  # Show first 5
        rel_docs = list(qrels[qid].keys())[:3]  # Show first 3 relevant docs
        logger.info(f"Query {qid} has {len(qrels[qid])} relevant docs (sample: {rel_docs})")

    return corpus, queries, dict(qrels)


def main():
    # File paths - UPDATE THESE WITH YOUR ACTUAL FILE PATHS
    corpus_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/corpus.tsv"
    queries_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/queries.tsv"
    qrels_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/qrels.tsv"

    try:
        # Load data
        logger.info("Loading data...")
        corpus, queries, qrels = load_tsv_data(corpus_file, queries_file, qrels_file)

        # Check if we have data
        if not corpus:
            logger.error("Error: No corpus data loaded!")
            return
        if not queries:
            logger.error("Error: No queries loaded!")
            return

        # Check qrels structure
        queries_with_judgments = [qid for qid in queries if qid in qrels and qrels[qid]]
        logger.info(f"Queries with relevant documents: {len(queries_with_judgments)}")

        if not queries_with_judgments:
            logger.error("No queries have relevance judgments!")
            logger.info("Available query IDs: " + str(list(queries.keys())))
            logger.info("Qrels query IDs: " + str(list(qrels.keys())))
            return

        # Initialize BioBERT evaluator with caching
        logger.info("Initializing BioBERT with caching...")
        start_time = time.time()

        evaluator = BioBERTEvaluatorWithCache(
            corpus, queries, qrels,
            model_name='dmis-lab/biobert-v1.1',
            cache_dir='./embedding_cache_biobert'
        )

        init_time = time.time()
        logger.info(f"Model initialization time: {init_time - start_time:.2f} seconds")

        # Evaluate
        logger.info("Evaluating BioBERT...")
        results = evaluator.evaluate(k=10, relevance_threshold=1.0, query_batch_size=64)

        end_time = time.time()

        # Print results
        print("\n" + "=" * 60)
        print("BIOBERT EVALUATION RESULTS")
        print("=" * 60)
        for metric, score in results.items():
            if metric != "num_queries" and metric != "relevance_threshold":
                print(f"{metric}: {score:.4f}")
            else:
                print(f"{metric}: {score}")
        print(f"Initialization time: {init_time - start_time:.2f} seconds")
        print(f"Evaluation time: {end_time - init_time:.2f} seconds")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"Error: File not found - {e}")
        logger.error("Please update the file paths in the main() function")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()