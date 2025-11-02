import numpy as np
from rank_bm25 import BM25Okapi
from collections import defaultdict
import time
from multiprocessing import Pool, cpu_count


class DebugBM25Evaluator:
    def __init__(self, corpus, queries, qrels, n_threads=None, relevance_threshold=1):
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels
        self.relevance_threshold = relevance_threshold

        # Preprocess and index corpus
        self.doc_ids = list(corpus.keys())
        self.doc_texts = list(corpus.values())

        print("Tokenizing corpus...")
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.doc_texts]

        print("Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.n_threads = n_threads or min(cpu_count(), 8)

    def _tokenize(self, text):
        return text.lower().split()

    def _debug_single_query(self, args):
        """
        Corrected MRR@k logic:
        - True MRR computed over *full ranking*
        - Contribution set to 0 if first relevant doc is beyond top-k
        """
        query_id, query_text, k, relevant_docs = args
        print(f"Running corrected MRR logic for query {query_id}")  # DEBUG

        tokenized_query = self._tokenize(query_text)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = np.argsort(scores)[::-1]
        ranked_doc_ids = [self.doc_ids[i] for i in ranked_indices]

        # ---- TRUE MRR@k ----
        mrr = 0.0
        for rank, doc_id in enumerate(ranked_doc_ids, start=1):
            if doc_id in relevant_docs:
                if rank <= k:
                    mrr = 1.0 / rank
                else:
                    mrr = 0.0  # beyond top-k → no MRR credit
                break

        # ---- Top-k Precision/Recall ----
        top_k_indices = ranked_indices[:k]
        top_k_doc_ids = [self.doc_ids[i] for i in top_k_indices]
        top_k_scores = [scores[i] for i in top_k_indices]

        retrieved_relevant = [doc_id for doc_id in top_k_doc_ids if doc_id in relevant_docs]

        precision_at_k = len(retrieved_relevant) / k
        recall_at_k = len(retrieved_relevant) / len(relevant_docs) if relevant_docs else 0.0

        return {
            "query_id": query_id,
            "mrr": mrr,
            "recall": recall_at_k,
            "precision": precision_at_k,
            "retrieved_relevant": retrieved_relevant,
            "top_k_docs": list(zip(top_k_doc_ids, top_k_scores)),
            "num_relevant": len(relevant_docs),
        }

    def evaluate(self, k=10, use_parallel=True):
        # Prepare query data
        query_data = []
        for query_id, query_text in self.queries.items():
            if query_id in self.qrels:
                query_qrels = self.qrels[query_id]
                relevant_docs = set(doc_id for doc_id, rel in query_qrels.items()
                                  if rel >= self.relevance_threshold)
                if relevant_docs:
                    query_data.append((query_id, query_text, k, relevant_docs))

        print(f"Evaluating {len(query_data)} queries...")

        if not query_data:
            return {
                f"MRR@{k}": 0.0, f"Recall@{k}": 0.0, f"Precision@{k}": 0.0,
                "num_queries": 0, "evaluation_time": 0.0
            }

        start_time = time.time()

        if use_parallel and len(query_data) > 1:
            with Pool(self.n_threads) as pool:
                results = list(pool.map(self._debug_single_query, query_data))
        else:
            results = [self._debug_single_query(data) for data in query_data]

        end_time = time.time()

        # Calculate aggregate metrics
        mrr_scores = [r['mrr'] for r in results]
        recall_scores = [r['recall'] for r in results]
        precision_scores = [r['precision'] for r in results]

        return {
            f"MRR@{k}": np.mean(mrr_scores),
            f"Recall@{k}": np.mean(recall_scores),
            f"Precision@{k}": np.mean(precision_scores),
            "num_queries": len(results),
            "evaluation_time": end_time - start_time,
            "detailed_results": results
        }


def load_tsv_data_fast(corpus_file, queries_file, qrels_file):
    """
    Faster TSV data loading with header handling
    """
    corpus = {}
    queries = {}
    qrels = defaultdict(dict)

    # Load corpus
    print(f"Loading corpus from {corpus_file}...")
    with open(corpus_file, 'r', encoding='utf-8', buffering=8192) as f:
        for i, line in enumerate(f):
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) == 2:
                doc_id, text = parts
                corpus[doc_id] = text
            if i % 100000 == 0 and i > 0:
                print(f"  Loaded {i:,} documents...")

    # Load queries - FIXED: Skip header properly
    print(f"Loading queries from {queries_file}...")
    with open(queries_file, 'r', encoding='utf-8', buffering=8192) as f:
        for i, line in enumerate(f):
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) == 2:
                query_id, query_text = parts
                # Skip header row
                if i == 0 and (query_id == 'query_id' or query_text == 'query'):
                    continue
                queries[query_id] = query_text

    # Load qrels
    print(f"Loading qrels from {qrels_file}...")
    with open(qrels_file, 'r', encoding='utf-8', buffering=8192) as f:
        first_line = True
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id, doc_id, relevance_str = parts[0], parts[1], parts[2]

                # Skip header row
                if first_line and (relevance_str.lower() == 'rel' or relevance_str.lower() == 'relevance'):
                    first_line = False
                    continue

                try:
                    relevance = float(relevance_str)
                    qrels[query_id][doc_id] = relevance
                except ValueError:
                    print(f"Warning: Skipping line {line_num} with invalid relevance '{relevance_str}'")
                    continue

    print(f"Corpus: {len(corpus):,} documents")
    print(f"Queries: {len(queries):,} queries")
    print(f"Qrels: {sum(len(docs) for docs in qrels.values()):,} relevance judgments")

    # Debug: Print actual queries loaded
    print("\nLoaded queries:")
    for query_id, query_text in queries.items():
        print(f"  {query_id}: '{query_text}'")

    return dict(corpus), dict(queries), dict(qrels)


def validate_data(corpus, queries, qrels):
    """Validate that the data makes sense"""
    print("\n" + "=" * 50)
    print("DATA VALIDATION")
    print("=" * 50)

    # Check if queries are meaningful
    for query_id, query_text in list(queries.items())[:3]:
        print(f"Query {query_id}: '{query_text}'")
        if len(query_text.split()) < 3:
            print(f"  WARNING: Query seems too short!")

    # Check query-document relationships
    print(f"\nQuery-document relationships:")
    for query_id in list(queries.keys())[:3]:
        if query_id in qrels:
            relevant_docs = qrels[query_id]
            highly_relevant = [doc for doc, rel in relevant_docs.items() if rel >= 1]
            print(f"  Query {query_id}: {len(highly_relevant)} highly relevant docs")

            # Check if any highly relevant documents exist in corpus
            missing_in_corpus = [doc for doc in highly_relevant if doc not in corpus]
            if missing_in_corpus:
                print(f"    WARNING: {len(missing_in_corpus)} relevant docs missing from corpus!")

    # Check BM25 scores for a sample query
    if queries:
        sample_query_id = list(queries.keys())[0]
        sample_query_text = queries[sample_query_id]
        print(f"\nSample query analysis for '{sample_query_text}':")

        # Simple word matching check
        query_words = set(sample_query_text.lower().split())
        print(f"  Query words: {query_words}")

        # Check a few relevant documents for this query
        if sample_query_id in qrels:
            relevant_docs = list(qrels[sample_query_id].keys())[:3]
            for doc_id in relevant_docs:
                if doc_id in corpus:
                    doc_text = corpus[doc_id][:100] + "..." if len(corpus[doc_id]) > 100 else corpus[doc_id]
                    print(f"  Relevant doc {doc_id}: {doc_text}")

def debug_data_relationships(corpus, queries, qrels):
    """Debug function to understand the relationships between queries and qrels"""
    print("\n" + "=" * 50)
    print("DATA RELATIONSHIP DEBUG INFO")
    print("=" * 50)

    # Check query IDs in both queries and qrels
    common_query_ids = set(queries.keys()) & set(qrels.keys())
    print(f"Common query IDs in both queries and qrels: {len(common_query_ids)}")

    # Check relevance score distribution
    relevance_scores = []
    for query_id, docs in qrels.items():
        for doc_id, rel in docs.items():
            relevance_scores.append(rel)

    if relevance_scores:
        unique_scores = set(relevance_scores)
        print(f"Unique relevance scores: {sorted(unique_scores)}")

        for score in sorted(unique_scores):
            count = sum(1 for rel in relevance_scores if rel == score)
            print(f"  Score {score}: {count} occurrences")

    # Check first few queries with their relevant documents
    print("\nFirst 3 queries with their relevant documents:")
    for i, query_id in enumerate(list(queries.keys())[:3]):
        if query_id in qrels:
            relevant_docs = qrels[query_id]
            highly_relevant = {doc: rel for doc, rel in relevant_docs.items() if rel >= 1}
            print(f"  Query {query_id}: {len(relevant_docs)} total relevant, {len(highly_relevant)} with rel>=1")
            if highly_relevant:
                print(f"    Highly relevant docs: {list(highly_relevant.items())[:3]}...")

# Keep your existing inspect_files function
def inspect_files(corpus_file, queries_file, qrels_file):
    """Inspect the first few lines of each file to understand the format"""
    print("Inspecting file formats...")

    for file_path, file_type in [(corpus_file, "corpus"), (queries_file, "queries"), (qrels_file, "qrels")]:
        print(f"\n--- First 3 lines of {file_type} file ---")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    print(f"Line {i + 1}: {repr(line.strip())}")
        except Exception as e:
            print(f"Error reading {file_type}: {e}")


def main():
    # File paths
    corpus_file = "/root/pycharm_semanticsearch/dataset/covid/test/corpus.tsv"
    queries_file = "/root/pycharm_semanticsearch/dataset/covid/test/queries.tsv"
    qrels_file = "/root/pycharm_semanticsearch/dataset/covid/test/qrels.tsv"

    try:
        # Load data
        corpus, queries, qrels = load_tsv_data_fast(corpus_file, queries_file, qrels_file)

        if not corpus or not queries:
            print("Error: No data loaded!")
            return

        # Validate data
        validate_data(corpus, queries, qrels)

        # Only proceed if queries look reasonable
        meaningful_queries = [q for q in queries.values() if len(q.split()) >= 3]
        if len(meaningful_queries) == 0:
            print("\nERROR: No meaningful queries found! Check your queries file format.")
            return

        print(f"\nProceeding with {len(queries)} queries...")

        # Initialize evaluator
        print("Initializing BM25...")
        evaluator = DebugBM25Evaluator(corpus, queries, qrels, n_threads=8, relevance_threshold=1)

        # Evaluate
        print("Evaluating BM25...")
        results = evaluator.evaluate(k=10, use_parallel=True)

        # Print results
        print("\n" + "=" * 60)
        print("BM25 EVALUATION RESULTS")
        print("=" * 60)
        for metric, score in results.items():
            if metric == "evaluation_time":
                print(f"Query processing time: {score:.2f} seconds")
            elif metric == "num_queries":
                print(f"{metric}: {score:,}")
            elif metric != "detailed_results":
                print(f"{metric}: {score:.4f}")

        # Check if results are reasonable
        if results["MRR@10"] > 0.9:
            print("\nWARNING: Suspiciously high MRR@10 - possible data issues!")
        if results["Precision@10"] > 0.8:
            print("WARNING: Suspiciously high Precision@10 - possible data issues!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


