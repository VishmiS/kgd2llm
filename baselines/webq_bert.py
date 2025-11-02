# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/baselines/webq_bert.py

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
from tqdm import tqdm
import faiss
import time
import os


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
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            encoded = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = self.model(**encoded)
            # Mean pooling over token embeddings
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())
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

    def evaluate(self, corpus, queries, qrels, k=10):
        self.doc_ids = list(corpus.keys())
        self.doc_texts = list(corpus.values())

        print("Encoding documents...")
        doc_embeddings = self._encode_texts(self.doc_texts)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(doc_embeddings)

        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
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



def main():
    # Update these paths
    corpus_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/corpus.tsv"
    queries_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/queries.tsv"
    qrels_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/qrels.tsv"

    corpus, queries, qrels = load_tsv_data(corpus_file, queries_file, qrels_file)

    evaluator = BERTEvaluatorFAISS(batch_size=32, device='cuda')
    start_time = time.time()
    results = evaluator.evaluate(corpus, queries, qrels, k=10)
    end_time = time.time()

    print("\n" + "="*60)
    print("BERT EVALUATION RESULTS")
    print("="*60)
    for metric, score in results.items():
        if metric != "num_queries":
            print(f"{metric}: {score:.4f}")
        else:
            print(f"{metric}: {score}")
    print(f"Evaluation time: {(end_time - start_time)/60:.2f} minutes")
    print("="*60)


if __name__ == "__main__":
    main()
