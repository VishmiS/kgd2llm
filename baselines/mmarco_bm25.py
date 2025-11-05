# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/baselines/mmarco_bm25.py

import numpy as np
import time
from collections import defaultdict
import os
import pickle
import hashlib
from tqdm import tqdm
from rank_bm25 import BM25Okapi
import re
from typing import Dict, List, Tuple, Union, Set
from multiprocessing import Pool, cpu_count


class BM25Evaluator:
    def __init__(self, corpus: Dict[str, str], queries: Dict[str, str], qrels: Dict[str, Dict[str, float]],
                 tokenizer_regex: str = r'(?u)\b\w\w+\b', n_threads: int = None):
        """
        BM25 evaluator with multiple optimization strategies
        """
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.tokenizer_regex = tokenizer_regex
        self.n_threads = n_threads or min(16, cpu_count() - 1)  # Limit to 16 threads to avoid overhead

        # Preprocess corpus and build BM25 index
        print("Building BM25 index...")
        self.doc_ids = list(corpus.keys())
        self.doc_texts = list(corpus.values())

        # Tokenize documents
        print("Tokenizing documents...")
        self.tokenized_docs = self._tokenize_corpus(self.doc_texts)

        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Create doc_id to index mapping for faster lookup
        self.doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}

        print(f"BM25 index built with {len(self.tokenized_docs)} documents")
        print(f"Using {self.n_threads} threads for parallel processing")

    def _tokenize_corpus(self, texts: List[str]) -> List[List[str]]:
        """Tokenize all documents in the corpus"""
        tokenized_docs = []
        for text in tqdm(texts, desc="Tokenizing documents"):
            tokens = re.findall(self.tokenizer_regex, text.lower())
            tokenized_docs.append(tokens)
        return tokenized_docs

    def _tokenize_query(self, query_text: str) -> List[str]:
        """Tokenize a single query"""
        return re.findall(self.tokenizer_regex, query_text.lower())

    def search(self, query_text: str, k: int = 10) -> List[Tuple[str, float]]:
        """Search for similar documents using BM25"""
        tokenized_query = self._tokenize_query(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k documents
        top_k_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_indices:
            doc_id = self.doc_ids[idx]
            score = scores[idx]
            results.append((doc_id, float(score)))

        return results

    def evaluate_optimized_batch(self, k: int = 10, relevance_threshold: float = 1.0,
                                 batch_size: int = 200) -> Dict[str, float]:
        """
        Optimized batch evaluation with better progress tracking
        """
        # Filter queries that have relevant documents
        query_data = []
        for query_id, query_text in self.queries.items():
            relevant_docs_dict = self.qrels.get(query_id, {})
            relevant_docs = set(doc_id for doc_id, rel_score in relevant_docs_dict.items()
                                if rel_score >= relevance_threshold)
            if relevant_docs:
                query_data.append((query_id, query_text, relevant_docs))

        print(f"Evaluating BM25 on {len(query_data)} queries with relevant documents...")
        print(f"Using optimized batch processing with batch size {batch_size}...")

        mrr_scores = []
        recall_scores = []
        precision_scores = []

        n_batches = (len(query_data) + batch_size - 1) // batch_size

        # Main progress bar
        pbar = tqdm(total=len(query_data), desc="Processing queries")

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(query_data))

            batch_data = query_data[start_idx:end_idx]

            # Process each query in the current batch
            for query_id, query_text, relevant_docs in batch_data:
                results = self.search(query_text, k=k)
                retrieved_docs = [doc_id for doc_id, score in results]

                # Calculate metrics
                mrr = self._calculate_mrr(retrieved_docs, relevant_docs)
                recall = self._calculate_recall(retrieved_docs, relevant_docs)
                precision = self._calculate_precision(retrieved_docs, relevant_docs)

                mrr_scores.append(mrr)
                recall_scores.append(recall)
                precision_scores.append(precision)

                pbar.update(1)
                pbar.set_postfix({
                    'batch': f"{batch_idx + 1}/{n_batches}",
                    'avg_mrr': f"{np.mean(mrr_scores):.3f}",
                    'avg_recall': f"{np.mean(recall_scores):.3f}"
                })

        pbar.close()

        return self._compile_results(mrr_scores, recall_scores, precision_scores, k, relevance_threshold)

    def evaluate_smart_parallel(self, k: int = 10, relevance_threshold: float = 1.0,
                                queries_per_process: int = 100) -> Dict[str, float]:
        """
        Smart parallel evaluation with optimal chunk sizing
        """
        # Prepare queries for parallel processing
        query_data = []
        for query_id, query_text in self.queries.items():
            relevant_docs_dict = self.qrels.get(query_id, {})
            relevant_docs = set(doc_id for doc_id, rel_score in relevant_docs_dict.items()
                                if rel_score >= relevance_threshold)
            if relevant_docs:
                query_data.append((query_id, query_text, relevant_docs))

        print(f"Evaluating BM25 on {len(query_data)} queries with relevant documents...")
        print(f"Using smart parallel processing with {self.n_threads} processes...")

        # Optimal chunk sizing - balance between overhead and parallelism
        optimal_chunk_size = max(50, len(query_data) // (self.n_threads * 2))
        chunks = [query_data[i:i + optimal_chunk_size] for i in range(0, len(query_data), optimal_chunk_size)]

        print(f"Split into {len(chunks)} chunks ({optimal_chunk_size} queries per chunk)")

        # Process with progress bar
        results = []
        with Pool(self.n_threads) as pool:
            # Use imap_unordered for better performance
            with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
                for chunk_result in pool.imap_unordered(self._process_chunk_fast,
                                                        [(chunk, k) for chunk in chunks]):
                    results.append(chunk_result)
                    pbar.update(1)
                    completed_queries = sum(len(chunk_mrr) for chunk_mrr, _, _ in results[:pbar.n])
                    pbar.set_postfix({
                        'queries_done': completed_queries,
                        'chunks_completed': pbar.n
                    })

        # Combine results
        mrr_scores = []
        recall_scores = []
        precision_scores = []

        for chunk_mrr, chunk_recall, chunk_precision in results:
            mrr_scores.extend(chunk_mrr)
            recall_scores.extend(chunk_recall)
            precision_scores.extend(chunk_precision)

        return self._compile_results(mrr_scores, recall_scores, precision_scores, k, relevance_threshold)

    def _process_chunk_fast(self, args):
        """Process a chunk of queries for fast parallel evaluation"""
        chunk, k = args
        chunk_mrr = []
        chunk_recall = []
        chunk_precision = []

        for query_id, query_text, relevant_docs in chunk:
            results = self.search(query_text, k=k)
            retrieved_docs = [doc_id for doc_id, score in results]

            mrr = self._calculate_mrr(retrieved_docs, relevant_docs)
            recall = self._calculate_recall(retrieved_docs, relevant_docs)
            precision = self._calculate_precision(retrieved_docs, relevant_docs)

            chunk_mrr.append(mrr)
            chunk_recall.append(recall)
            chunk_precision.append(precision)

        return chunk_mrr, chunk_recall, chunk_precision

    def evaluate_sequential_with_progress(self, k: int = 10, relevance_threshold: float = 1.0) -> Dict[str, float]:
        """
        Sequential evaluation with detailed progress tracking
        """
        # Filter queries that have relevant documents
        query_data = []
        for query_id, query_text in self.queries.items():
            relevant_docs_dict = self.qrels.get(query_id, {})
            relevant_docs = set(doc_id for doc_id, rel_score in relevant_docs_dict.items()
                                if rel_score >= relevance_threshold)
            if relevant_docs:
                query_data.append((query_id, query_text, relevant_docs))

        print(f"Evaluating BM25 on {len(query_data)} queries with relevant documents...")
        print("Using sequential evaluation with progress tracking...")

        mrr_scores = []
        recall_scores = []
        precision_scores = []

        # Progress bar with detailed information
        pbar = tqdm(query_data, desc="Evaluating queries")

        for query_id, query_text, relevant_docs in pbar:
            results = self.search(query_text, k=k)
            retrieved_docs = [doc_id for doc_id, score in results]

            # Calculate metrics
            mrr = self._calculate_mrr(retrieved_docs, relevant_docs)
            recall = self._calculate_recall(retrieved_docs, relevant_docs)
            precision = self._calculate_precision(retrieved_docs, relevant_docs)

            mrr_scores.append(mrr)
            recall_scores.append(recall)
            precision_scores.append(precision)

            # Update progress bar with current statistics
            if len(mrr_scores) % 100 == 0:  # Update every 100 queries to avoid overhead
                pbar.set_postfix({
                    'avg_mrr': f"{np.mean(mrr_scores):.3f}",
                    'avg_recall': f"{np.mean(recall_scores):.3f}",
                    'completed': f"{len(mrr_scores)}/{len(query_data)}"
                })

        pbar.close()

        return self._compile_results(mrr_scores, recall_scores, precision_scores, k, relevance_threshold)

    def evaluate(self, k: int = 10, relevance_threshold: float = 1.0,
                 method: str = 'optimized_batch', **kwargs) -> Dict[str, float]:
        """
        Main evaluation method with choice of optimization strategy
        """
        print(f"\nStarting evaluation with method: {method}, k={k}")
        start_time = time.time()

        if method == 'sequential':
            result = self.evaluate_sequential_with_progress(k, relevance_threshold)
        elif method == 'optimized_batch':
            batch_size = kwargs.get('batch_size', 200)
            result = self.evaluate_optimized_batch(k, relevance_threshold, batch_size)
        elif method == 'smart_parallel':
            result = self.evaluate_smart_parallel(k, relevance_threshold)
        else:
            raise ValueError(f"Unknown evaluation method: {method}")

        eval_time = time.time() - start_time
        result['evaluation_time'] = eval_time
        print(f"Evaluation completed in {eval_time:.2f} seconds")

        return result

    def _compile_results(self, mrr_scores: List[float], recall_scores: List[float],
                         precision_scores: List[float], k: int, relevance_threshold: float) -> Dict[str, float]:
        """Compile final results from metric scores"""
        if mrr_scores:
            final_mrr = np.mean(mrr_scores)
            final_recall = np.mean(recall_scores)
            final_precision = np.mean(precision_scores)
        else:
            final_mrr = 0.0
            final_recall = 0.0
            final_precision = 0.0
            print("Warning: No queries with relevant documents found!")

        return {
            f"MRR@{k}": final_mrr,
            f"Recall@{k}": final_recall,
            f"Precision@{k}": final_precision,
            "num_queries": len(mrr_scores),
            "relevance_threshold": relevance_threshold
        }

    def _calculate_mrr(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """Calculate MRR for a single query"""
        for rank, doc_id in enumerate(retrieved_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / rank
        return 0.0

    def _calculate_recall(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """Calculate Recall for a single query"""
        if not relevant_docs:
            return 0.0
        retrieved_relevant = len([doc for doc in retrieved_docs if doc in relevant_docs])
        return retrieved_relevant / len(relevant_docs)

    def _calculate_precision(self, retrieved_docs: List[str], relevant_docs: Set[str]) -> float:
        """Calculate Precision for a single query"""
        if not retrieved_docs:
            return 0.0
        retrieved_relevant = len([doc for doc in retrieved_docs if doc in relevant_docs])
        return retrieved_relevant / len(retrieved_docs)

    def get_query_statistics(self, relevance_threshold: float = 1.0) -> Dict[str, any]:
        """Get statistics about queries and relevance judgments"""
        stats = {
            'total_queries': len(self.queries),
            'queries_with_relevant': 0,
            'total_relevance_judgments': 0,
            'relevance_score_distribution': defaultdict(int),
            'queries_by_relevant_count': defaultdict(int)
        }

        for query_id, rel_dict in self.qrels.items():
            relevant_docs = [doc_id for doc_id, score in rel_dict.items() if score >= relevance_threshold]
            if relevant_docs:
                stats['queries_with_relevant'] += 1
                stats['queries_by_relevant_count'][len(relevant_docs)] += 1

            for score in rel_dict.values():
                stats['relevance_score_distribution'][score] += 1
                stats['total_relevance_judgments'] += 1

        return stats


def load_tsv_data(corpus_file: str, queries_file: str, qrels_file: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load TSV files with support for float relevance scores
    """
    corpus = {}
    queries = {}
    qrels = defaultdict(dict)

    # Load corpus (format: doc_id\ttext)
    print(f"Loading corpus from {corpus_file}...")
    with open(corpus_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Skip header row (first line)
            if line_num == 1:
                print(f"Skipping header row in corpus: {line.strip()}")
                continue
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
            # Skip header row (first line)
            if line_num == 1:
                print(f"Skipping header row in queries: {line.strip()}")
                continue
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

    # Load qrels (format: query_id\tdoc_id\trelevance) - support both int and float
    print(f"Loading qrels from {qrels_file}...")
    with open(qrels_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Skip header row (first line)
            if line_num == 1:
                print(f"Skipping header row in qrels: {line.strip()}")
                continue
            try:
                parts = line.strip().split()
                if len(parts) >= 3:
                    query_id = parts[0]
                    doc_id = parts[1]

                    # Try to parse as float first, then as int
                    try:
                        relevance = float(parts[2])
                    except ValueError:
                        # If float fails, try int
                        relevance = int(parts[2])

                    qrels[query_id][doc_id] = relevance
                else:
                    print(f"Warning: Skipping malformed line {line_num} in qrels: {line.strip()}")
            except Exception as e:
                print(f"Error processing line {line_num} in qrels: {e}")

    print(f"Corpus: {len(corpus)} documents")
    print(f"Queries: {len(queries)} queries")
    print(f"Qrels: {sum(len(docs) for docs in qrels.values())} relevance judgments")

    # Print some statistics about relevance scores
    if qrels:
        all_relevance_scores = []
        for query_docs in qrels.values():
            all_relevance_scores.extend(query_docs.values())

        print(f"Relevance score range: {min(all_relevance_scores)} to {max(all_relevance_scores)}")
        print(f"Unique relevance scores: {sorted(set(all_relevance_scores))}")

    return dict(corpus), dict(queries), dict(qrels)


def print_evaluation_results(results: Dict[str, float], method: str, eval_time: float):
    """Print evaluation results in a formatted table"""
    print("\n" + "=" * 80)
    print("BM25 EVALUATION RESULTS")
    print("=" * 80)
    print(f"Evaluation method: {method}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Number of queries: {results['num_queries']}")
    print(f"Relevance threshold: {results['relevance_threshold']}")
    print("-" * 80)

    for metric, score in results.items():
        if metric not in ['num_queries', 'relevance_threshold', 'evaluation_time']:
            print(f"{metric:<15}: {score:.4f}")

    print("=" * 80)


def main():
    # File paths - UPDATE THESE WITH YOUR ACTUAL FILE PATHS
    corpus_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/corpus.tsv"
    queries_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/queries.tsv"
    qrels_file = "/root/pycharm_semanticsearch/dataset/ms_marco/test/qrels.tsv"

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

        # Initialize BM25 evaluator
        print("\n" + "=" * 60)
        print("INITIALIZING BM25 EVALUATOR")
        print("=" * 60)

        start_time = time.time()
        evaluator = BM25Evaluator(corpus, queries, qrels)
        init_time = time.time() - start_time
        print(f"BM25 initialization time: {init_time:.2f} seconds")

        # Get query statistics
        stats = evaluator.get_query_statistics()
        print(f"\nQuery Statistics:")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  Queries with relevant documents: {stats['queries_with_relevant']}")
        print(f"  Total relevance judgments: {stats['total_relevance_judgments']}")
        print(f"  Relevance score distribution: {dict(stats['relevance_score_distribution'])}")

        # Test different evaluation methods
        k_values = [10]

        for k in k_values:
            print(f"\n{'=' * 60}")
            print(f"EVALUATING AT k={k}")
            print(f"{'=' * 60}")

            # Try optimized batch first (most reliable)
            print(f"\nUsing optimized batch method...")
            try:
                results = evaluator.evaluate(k=k, relevance_threshold=1.0, method='optimized_batch', batch_size=100)
                print_evaluation_results(results, 'optimized_batch', results.get('evaluation_time', 0))
            except Exception as e:
                print(f"Optimized batch failed: {e}")
                print("Falling back to sequential method...")

                # Fall back to sequential method
                results = evaluator.evaluate(k=k, relevance_threshold=1.0, method='sequential')
                print_evaluation_results(results, 'sequential', results.get('evaluation_time', 0))

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please update the file paths in the main() function")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()