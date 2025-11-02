# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/baselines/covid_biobert.py

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm
import faiss
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BioBERTEvaluator:
    """BioBERT evaluation using FAISS for fast retrieval"""

    def __init__(self, model_name='dmis-lab/biobert-v1.1', batch_size=32, device=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"Loading BioBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _encode_texts(self, texts, description="Encoding"):
        """Encode texts using simple mean pooling with progress bar"""
        embeddings = []

        # Create progress bar for encoding
        pbar = tqdm(total=len(texts), desc=description, unit="text")

        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            outputs = self.model(**encoded)

            # Simple mean pooling
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())

            # Update progress bar
            pbar.update(len(batch_texts))

        pbar.close()

        embeddings = np.vstack(embeddings)

        # Normalize with FAISS
        faiss.normalize_L2(embeddings)
        return embeddings

    def evaluate(self, corpus, queries, qrels, k=10):
        """Evaluate BioBERT on MRR@k and Recall@k"""
        self.doc_ids = list(corpus.keys())
        self.doc_texts = list(corpus.values())

        logger.info("Encoding documents...")
        doc_embeddings = self._encode_texts(self.doc_texts, "Encoding documents")

        # Create FAISS index
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(doc_embeddings.astype('float32'))

        mrr_scores, recall_scores = [], []

        # Filter queries that have relevance judgments
        valid_queries = {qid: query for qid, query in queries.items()
                         if qid in qrels and qrels[qid]}

        if not valid_queries:
            logger.error("No queries with relevance judgments found!")
            return {
                f"MRR@{k}": 0.0,
                f"Recall@{k}": 0.0,
                "num_queries": 0
            }

        logger.info(f"Evaluating {len(valid_queries)} queries with relevance judgments...")

        # Encode all queries at once with progress bar
        query_ids = list(valid_queries.keys())
        query_texts = list(valid_queries.values())

        logger.info("Encoding queries...")
        query_embeddings = self._encode_texts(query_texts, "Encoding queries")

        # Search for each query
        for i, (query_id, query_embedding) in enumerate(zip(query_ids, query_embeddings)):
            relevant_docs = set(qrels[query_id].keys())

            # Reshape for single query
            query_embedding = query_embedding.reshape(1, -1)

            similarities, indices = index.search(query_embedding.astype('float32'), k)
            top_k_doc_ids = [self.doc_ids[idx] for idx in indices[0]]

            mrr = self._calculate_mrr(top_k_doc_ids, relevant_docs)
            recall = self._calculate_recall(top_k_doc_ids, relevant_docs)

            mrr_scores.append(mrr)
            recall_scores.append(recall)

        return {
            f"MRR@{k}": np.mean(mrr_scores),
            f"Recall@{k}": np.mean(recall_scores),
            "num_queries": len(mrr_scores)
        }

    def _calculate_mrr(self, retrieved_ids, relevant_ids):
        """Calculate MRR for a single query"""
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def _calculate_recall(self, retrieved_ids, relevant_ids):
        """Calculate Recall for a single query"""
        hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
        return hits / len(relevant_ids) if relevant_ids else 0.0


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
    logger.info(f"Query IDs: {list(queries.keys())}")
    queries_with_judgments = [qid for qid in queries if qid in qrels and qrels[qid]]
    logger.info(f"Queries with relevance judgments: {queries_with_judgments}")

    # Show some statistics about relevance judgments per query
    for qid in queries_with_judgments[:5]:  # Show first 5
        rel_docs = list(qrels[qid].keys())[:3]  # Show first 3 relevant docs
        logger.info(f"Query {qid} has {len(qrels[qid])} relevant docs (sample: {rel_docs})")

    return corpus, queries, dict(qrels)


def main():
    # File paths
    corpus_file = "/root/pycharm_semanticsearch/dataset/covid/test/corpus.tsv"
    queries_file = "/root/pycharm_semanticsearch/dataset/covid/test/queries.tsv"
    qrels_file = "/root/pycharm_semanticsearch/dataset/covid/test/qrels.tsv"

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

    # Initialize and evaluate BioBERT
    logger.info("Initializing BioBERT evaluator...")
    evaluator = BioBERTEvaluator(batch_size=32, device='cuda')

    start_time = time.time()
    results = evaluator.evaluate(corpus, queries, qrels, k=10)
    end_time = time.time()

    # Print results
    print("\n" + "=" * 60)
    print("BIOBERT EVALUATION RESULTS")
    print("=" * 60)
    for metric, score in results.items():
        if metric != "num_queries":
            print(f"{metric}: {score:.4f}")
        else:
            print(f"{metric}: {score}")
    print(f"Evaluation time: {(end_time - start_time) / 60:.2f} minutes")
    print("=" * 60)


if __name__ == "__main__":
    main()