# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/baselines/webq_biobert.py

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
    def _encode_texts(self, texts):
        """Encode texts using simple mean pooling"""
        embeddings = []
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

        embeddings = np.vstack(embeddings)

        # Normalize with FAISS
        faiss.normalize_L2(embeddings)
        return embeddings

    def evaluate(self, corpus, queries, qrels, k=10):
        """Evaluate BioBERT on MRR@k and Recall@k"""
        self.doc_ids = list(corpus.keys())
        self.doc_texts = list(corpus.values())

        logger.info("Encoding documents...")
        doc_embeddings = self._encode_texts(self.doc_texts)

        # Create FAISS index
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(doc_embeddings.astype('float32'))

        mrr_scores, recall_scores = [], []

        logger.info("Evaluating queries...")
        for query_id, query_text in tqdm(queries.items()):
            relevant_docs = set(qrels.get(query_id, {}).keys())
            if not relevant_docs:
                continue

            query_embedding = self._encode_texts([query_text])

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
    """Load TSV files for corpus, queries, and qrels"""
    corpus, queries, qrels = {}, {}, defaultdict(dict)

    # Load corpus
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                corpus[parts[0]] = '\t'.join(parts[1:])

    # Load queries
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = '\t'.join(parts[1:])

    # Load qrels
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    relevance = int(parts[2])
                    qrels[parts[0]][parts[1]] = relevance
                except ValueError:
                    continue

    logger.info(
        f"Loaded: {len(corpus)} documents, {len(queries)} queries, {sum(len(d) for d in qrels.values())} relevance judgments")
    return corpus, queries, dict(qrels)


def main():
    # File paths
    corpus_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/corpus.tsv"
    queries_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/queries.tsv"
    qrels_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/qrels.tsv"

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

    logger.info(f"Queries with relevant documents: {sum(1 for qid in queries if qid in qrels and qrels[qid])}")

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