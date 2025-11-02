# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/baselines/covid_con.py

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import normalize
import json
import time
from collections import defaultdict
import faiss


class ContrieverEvaluator:
    def __init__(self, corpus, queries, qrels, model_name="facebook/contriever"):
        """
        Initialize Contriever evaluator

        Args:
            corpus: dict with doc_id as key and document text as value
            queries: dict with query_id as key and query text as value
            qrels: dict with query_id as key and dict of relevant doc_ids as value
            model_name: pre-trained Contriever model name
        """
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels

        # Load Contriever model and tokenizer
        print(f"Loading Contriever model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Preprocess and encode corpus
        self.doc_ids = list(corpus.keys())
        self.doc_texts = list(corpus.values())

        print("Encoding corpus...")
        self.doc_embeddings = self._encode_corpus(self.doc_texts)

        # Build FAISS index for efficient similarity search
        print("Building FAISS index...")
        self.index = self._build_faiss_index(self.doc_embeddings)

    def _encode_corpus(self, texts, batch_size=32):
        """Encode all documents in the corpus"""
        all_embeddings = []

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

            if (i // batch_size) % 10 == 0:
                print(f"Encoded {i}/{len(texts)} documents...")

        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)

        # Normalize embeddings for cosine similarity
        all_embeddings = normalize(all_embeddings, axis=1, norm='l2')

        return all_embeddings

    def _mean_pooling(self, token_embeddings, attention_mask):
        """Apply mean pooling to token embeddings with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _build_faiss_index(self, embeddings):
        """Build FAISS index for efficient similarity search"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings.astype(np.float32))
        return index

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

    def evaluate(self, k=10):
        """
        Evaluate Contriever on MRR@k and Recall@k

        Args:
            k: cutoff for evaluation metrics

        Returns:
            dict with MRR@k and Recall@k scores
        """
        mrr_scores = []
        recall_scores = []

        for query_id, query_text in self.queries.items():
            # Get relevant documents for this query
            relevant_docs = set(self.qrels.get(query_id, {}).keys())

            if not relevant_docs:
                continue  # Skip queries with no relevant documents

            # Search using Contriever
            results = self.search(query_text, k=k)
            top_k_doc_ids = [doc_id for doc_id, score in results]

            # Calculate MRR@k
            mrr = self._calculate_mrr(top_k_doc_ids, relevant_docs)
            mrr_scores.append(mrr)

            # Calculate Recall@k
            recall = self._calculate_recall(top_k_doc_ids, relevant_docs)
            recall_scores.append(recall)

            # Print progress for each query
            print(f"Query: {query_id} | MRR@{k}: {mrr:.4f} | Recall@{k}: {recall:.4f}")

        # Calculate final metrics
        final_mrr = np.mean(mrr_scores)
        final_recall = np.mean(recall_scores)

        return {
            f"MRR@{k}": final_mrr,
            f"Recall@{k}": final_recall,
            "num_queries": len(mrr_scores)
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
    print(f"Loading corpus from {corpus_file}...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    doc_id = parts[0]
                    text = '\t'.join(parts[1:])  # Handle cases where text contains tabs
                    corpus[doc_id] = text
                else:
                    print(f"Warning: Skipping malformed line {line_num} in corpus: {line.strip()}")
            except Exception as e:
                print(f"Error processing line {line_num} in corpus: {e}")

    # Load queries (format: query_id\tquery_text)
    print(f"Loading queries from {queries_file}...")
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    query_id = parts[0]
                    query_text = '\t'.join(parts[1:])
                    queries[query_id] = query_text
                else:
                    print(f"Warning: Skipping malformed line {line_num} in queries: {line.strip()}")
            except Exception as e:
                print(f"Error processing line {line_num} in queries: {e}")

    # Load qrels (format: query_id\tdoc_id\trelevance)
    print(f"Loading qrels from {qrels_file}...")
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
                    print(f"Warning: Skipping malformed line {line_num} in qrels: {line.strip()}")
            except Exception as e:
                print(f"Error processing line {line_num} in qrels: {e}")

    print(f"Corpus: {len(corpus)} documents")
    print(f"Queries: {len(queries)} queries")
    print(f"Qrels: {sum(len(docs) for docs in qrels.values())} relevance judgments")

    return dict(corpus), dict(queries), dict(qrels)


def main():
    # File paths - UPDATE THESE WITH YOUR ACTUAL FILE PATHS
    corpus_file = "/root/pycharm_semanticsearch/dataset/covid/test/corpus.tsv"
    queries_file = "/root/pycharm_semanticsearch/dataset/covid/test/queries.tsv"
    qrels_file = "/root/pycharm_semanticsearch/dataset/covid/test/qrels.tsv"

    try:
        # Load data using TSV loader
        print("Loading data...")
        corpus, queries, qrels = load_tsv_data(corpus_file, queries_file, qrels_file)

        # Check if we have any data
        if not corpus:
            print("Error: No corpus data loaded!")
            return
        if not queries:
            print("Error: No queries loaded!")
            return
        if not qrels:
            print("Warning: No qrels loaded!")

        print(f"Loaded {len(corpus)} documents, {len(queries)} queries, {len(qrels)} query-judgment pairs")

        # Check for queries that have relevant documents
        queries_with_relevant = [qid for qid in queries if qid in qrels and qrels[qid]]
        print(f"Queries with relevant documents: {len(queries_with_relevant)}")

        # Initialize Contriever evaluator
        print("Initializing Contriever...")
        evaluator = ContrieverEvaluator(corpus, queries, qrels)

        # Evaluate
        print("Evaluating Contriever...")
        start_time = time.time()

        results = evaluator.evaluate(k=10)

        end_time = time.time()

        # Print results
        print("\n" + "=" * 50)
        print("CONTRIEVER EVALUATION RESULTS")
        print("=" * 50)
        for metric, score in results.items():
            if metric != "num_queries":
                print(f"{metric}: {score:.4f}")
            else:
                print(f"{metric}: {score}")
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please update the file paths in the main() function")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()