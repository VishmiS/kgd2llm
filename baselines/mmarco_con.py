# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/baselines/mmarco_con.py

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


class ContrieverEvaluatorWithCache:
    def __init__(self, corpus, queries, qrels, model_name="facebook/contriever",
                 cache_dir='./embedding_cache_contriever'):
        """
        Initialize Contriever evaluator with caching

        Args:
            corpus: dict with doc_id as key and document text as value
            queries: dict with query_id as key and query text as value
            qrels: dict with query_id as key and dict of relevant doc_ids as value
            model_name: pre-trained Contriever model name
            cache_dir: directory to cache embeddings
        """
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.cache_dir = cache_dir
        self.model_name = model_name

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Load Contriever model and tokenizer
        logger.info(f"Loading Contriever model: {model_name}")
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

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to token embeddings with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

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

            # Encode with Contriever
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Contriever uses mean pooling of last hidden states
                embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = embeddings.cpu().numpy()
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
            query_embedding = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
            query_embedding = query_embedding.cpu().numpy()

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
                batch_embeddings = self._mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
                batch_embeddings = batch_embeddings.cpu().numpy()
                all_embeddings.append(batch_embeddings)

        # Concatenate and normalize
        all_embeddings = np.vstack(all_embeddings)
        all_embeddings = normalize(all_embeddings, axis=1, norm='l2')

        return all_embeddings.astype(np.float32)

    def search(self, query_text, k=10):
        """Search for similar documents using Contriever"""
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
        Evaluate Contriever on MRR@k and Recall@k with optimized query processing

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
    """
    Load TSV files
    """
    corpus = {}
    queries = {}
    qrels = defaultdict(dict)

    # Load corpus (format: doc_id\ttext)
    logger.info(f"Loading corpus from {corpus_file}...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    doc_id = parts[0]
                    text = '\t'.join(parts[1:])  # Handle cases where text contains tabs
                    corpus[doc_id] = text
                else:
                    logger.warning(f"Skipping malformed line {line_num} in corpus: {line.strip()}")
            except Exception as e:
                logger.error(f"Error processing line {line_num} in corpus: {e}")

    # Load queries (format: query_id\tquery_text)
    logger.info(f"Loading queries from {queries_file}...")
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    query_id = parts[0]
                    query_text = '\t'.join(parts[1:])
                    queries[query_id] = query_text
                else:
                    logger.warning(f"Skipping malformed line {line_num} in queries: {line.strip()}")
            except Exception as e:
                logger.error(f"Error processing line {line_num} in queries: {e}")

    # Load qrels (format: query_id\tdoc_id\trelevance)
    logger.info(f"Loading qrels from {qrels_file}...")
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split()
                if len(parts) >= 3:
                    query_id = parts[0]
                    doc_id = parts[1]
                    # Handle both integer and float relevance scores
                    relevance_str = parts[2]
                    if '.' in relevance_str:
                        relevance = float(relevance_str)
                    else:
                        relevance = int(relevance_str)
                    qrels[query_id][doc_id] = relevance
                else:
                    logger.warning(f"Skipping malformed line {line_num} in qrels: {line.strip()}")
            except Exception as e:
                logger.error(f"Error processing line {line_num} in qrels: {e}")

    logger.info(f"Corpus: {len(corpus)} documents")
    logger.info(f"Queries: {len(queries)} queries")
    logger.info(f"Qrels: {sum(len(docs) for docs in qrels.values())} relevance judgments")

    return dict(corpus), dict(queries), dict(qrels)


def main():
    # File paths - UPDATE THESE WITH YOUR ACTUAL FILE PATHS
    corpus_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/corpus.tsv"
    queries_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/queries.tsv"
    qrels_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/qrels.tsv"

    try:
        # Load data using TSV loader
        logger.info("Loading data...")
        corpus, queries, qrels = load_tsv_data(corpus_file, queries_file, qrels_file)

        # Check if we have any data
        if not corpus:
            logger.error("Error: No corpus data loaded!")
            return
        if not queries:
            logger.error("Error: No queries loaded!")
            return
        if not qrels:
            logger.warning("Warning: No qrels loaded!")

        logger.info(f"Loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} query-judgment pairs")

        # Check for queries that have relevant documents
        queries_with_relevant = [qid for qid in queries if qid in qrels and qrels[qid]]
        logger.info(f"Queries with relevant documents: {len(queries_with_relevant)}")

        if not queries_with_relevant:
            logger.error("No queries have relevance judgments!")
            return

        # Initialize Contriever evaluator with caching
        logger.info("Initializing Contriever with caching...")
        start_time = time.time()

        evaluator = ContrieverEvaluatorWithCache(
            corpus, queries, qrels,
            model_name="facebook/contriever",
            cache_dir='./embedding_cache_contriever'
        )

        init_time = time.time()
        logger.info(f"Model initialization time: {init_time - start_time:.2f} seconds")

        # Evaluate
        logger.info("Evaluating Contriever...")
        results = evaluator.evaluate(k=10, relevance_threshold=1.0, query_batch_size=64)

        end_time = time.time()

        # Print results
        print("\n" + "=" * 60)
        print("CONTRIEVER EVALUATION RESULTS")
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