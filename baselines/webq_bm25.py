import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict
import json
import time


class BM25Evaluator:
    def __init__(self, corpus, queries, qrels):
        """
        Initialize BM25 evaluator

        Args:
            corpus: dict with doc_id as key and document text as value
            queries: dict with query_id as key and query text as value
            qrels: dict with query_id as key and dict of relevant doc_ids as value
        """
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels

        # Preprocess and index corpus
        self.doc_ids = list(corpus.keys())
        self.doc_texts = list(corpus.values())

        # Tokenize documents
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.doc_texts]

        # Initialize BM25
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text):
        """Simple tokenization function"""
        return text.lower().split()

    def evaluate(self, k=10):
        """
        Evaluate BM25 on MRR@k and Recall@k

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

            # Tokenize query
            tokenized_query = self._tokenize(query_text)

            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)

            # Get top-k document indices
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_doc_ids = [self.doc_ids[idx] for idx in top_k_indices]

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
                    relevance = int(parts[2])
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
    corpus_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/corpus.tsv"
    queries_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/queries.tsv"
    qrels_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/qrels.tsv"

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

        # Initialize evaluator
        print("Initializing BM25...")
        evaluator = BM25Evaluator(corpus, queries, qrels)

        # Evaluate
        print("Evaluating BM25...")
        start_time = time.time()

        results = evaluator.evaluate(k=10)

        end_time = time.time()

        # Print results
        print("\n" + "=" * 50)
        print("BM25 EVALUATION RESULTS")
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