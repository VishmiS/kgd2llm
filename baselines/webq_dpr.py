# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/baselines/webq_dpr.py

import numpy as np
import torch
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
from sklearn.preprocessing import normalize
import json
import time
from collections import defaultdict
import faiss
import pandas as pd
import os
from datetime import datetime
import re
import hashlib
import random
from tqdm import tqdm
import csv


class DPREvaluator:
    def __init__(self, corpus, queries, qrels,
                 question_model_name="facebook/dpr-question_encoder-single-nq-base",
                 context_model_name="facebook/dpr-ctx_encoder-single-nq-base"):
        """
        Initialize DPR evaluator

        Args:
            corpus: dict with doc_id as key and document text as value
            queries: dict with query_id as key and query text as value
            qrels: dict with query_id as key and dict of relevant doc_ids as value
            question_model_name: pre-trained DPR question encoder model name
            context_model_name: pre-trained DPR context encoder model name
        """
        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels

        # Load DPR models and tokenizers
        print(f"Loading DPR question encoder: {question_model_name}")
        self.question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(question_model_name)
        self.question_encoder = DPRQuestionEncoder.from_pretrained(question_model_name)

        print(f"Loading DPR context encoder: {context_model_name}")
        self.context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(context_model_name)
        self.context_encoder = DPRContextEncoder.from_pretrained(context_model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.question_encoder.to(self.device)
        self.context_encoder.to(self.device)
        self.question_encoder.eval()
        self.context_encoder.eval()

        # Preprocess and encode corpus
        self.doc_ids = list(corpus.keys())
        self.doc_texts = list(corpus.values())

        print("Encoding corpus with DPR context encoder...")
        self.doc_embeddings = self._encode_corpus(self.doc_texts)

        # Build FAISS index for efficient similarity search
        print("Building FAISS index...")
        self.index = self._build_faiss_index(self.doc_embeddings)

    def _encode_corpus(self, texts, batch_size=32):
        """Encode all documents in the corpus using DPR context encoder"""
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize batch with context tokenizer
            inputs = self.context_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Encode with context encoder
            with torch.no_grad():
                outputs = self.context_encoder(**inputs)
                # Use the pooler output as document representation
                embeddings = outputs.pooler_output.cpu().numpy()
                all_embeddings.append(embeddings)

            if (i // batch_size) % 10 == 0:
                print(f"Encoded {i}/{len(texts)} documents...")

        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)

        # Normalize embeddings for cosine similarity
        all_embeddings = normalize(all_embeddings, axis=1, norm='l2')

        return all_embeddings

    def _build_faiss_index(self, embeddings):
        """Build FAISS index for efficient similarity search"""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        index.add(embeddings.astype(np.float32))
        return index

    def _encode_query(self, query_text):
        """Encode a single query using DPR question encoder"""
        inputs = self.question_tokenizer(
            query_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.question_encoder(**inputs)
            query_embedding = outputs.pooler_output.cpu().numpy()

        # Normalize for cosine similarity
        query_embedding = normalize(query_embedding, axis=1, norm='l2')

        return query_embedding.astype(np.float32)

    def search(self, query_text, k=10):
        """Search for similar documents using DPR"""
        query_embedding = self._encode_query(query_text)

        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, k)

        # Convert to document IDs and scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            doc_id = self.doc_ids[idx]
            results.append((doc_id, float(score)))

        return results

    def evaluate(self, k=10, max_queries=None):
        """
        Evaluate DPR on MRR@k and Recall@k with detailed results

        Args:
            k: cutoff for evaluation metrics
            max_queries: maximum number of queries to evaluate

        Returns:
            tuple: (metrics_dict, detailed_results_list)
        """
        mrr_scores = []
        recall_scores = []
        detailed_results = []

        query_items = list(self.queries.items())
        if max_queries:
            query_items = query_items[:max_queries]

        for query_id, query_text in tqdm(query_items, desc="Evaluating queries"):
            # Get relevant documents for this query - qrels[query_id] is a dict, we need the keys
            relevant_docs = set(self.qrels.get(query_id, {}).keys())

            if not relevant_docs:
                continue  # Skip queries with no relevant documents

            # Search using DPR
            results = self.search(query_text, k=k)
            top_k_doc_ids = [doc_id for doc_id, score in results]
            top_k_scores = [score for doc_id, score in results]

            # Calculate MRR@k
            mrr = self._calculate_mrr(top_k_doc_ids, relevant_docs)
            mrr_scores.append(mrr)

            # Calculate Recall@k
            recall = self._calculate_recall(top_k_doc_ids, relevant_docs)
            recall_scores.append(recall)

            # Generate detailed result
            detailed_result = self._generate_detailed_result(
                query_id, query_text, relevant_docs, top_k_doc_ids, top_k_scores, mrr, recall
            )
            detailed_results.append(detailed_result)

        # Calculate final metrics
        final_mrr = np.mean(mrr_scores) if mrr_scores else 0.0
        final_recall = np.mean(recall_scores) if recall_scores else 0.0

        metrics = {
            f"MRR@{k}": final_mrr,
            f"Recall@{k}": final_recall,
            "num_queries": len(mrr_scores)
        }

        return metrics, detailed_results

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

    def _generate_detailed_result(self, query_id, query_text, relevant_docs, top_k_doc_ids, top_k_scores, mrr, recall):
        """Generate detailed result for a single query"""
        # Find first relevant rank
        first_relevant_rank = None
        for rank, doc_id in enumerate(top_k_doc_ids, 1):
            if doc_id in relevant_docs:
                first_relevant_rank = rank
                break

        # Generate explanation
        if first_relevant_rank is None:
            explanation = "No relevant documents found in top results."
        elif first_relevant_rank == 1:
            explanation = "First relevant document found at rank 1."
        else:
            explanation = f"First relevant document found at rank {first_relevant_rank}."

        # Get correct answers from relevant documents
        correct_answers = self._get_correct_answers(relevant_docs)

        # Get top documents sample
        top_docs_sample = self._get_top_docs_sample(top_k_doc_ids)

        return {
            'query_id': query_id,
            'query_text': query_text,
            'relevant_doc_ids': list(relevant_docs),
            'top_k_doc_ids': top_k_doc_ids,
            'top_k_scores': top_k_scores,
            'first_relevant_rank': first_relevant_rank,
            'MRR': mrr,
            'Recall': recall,
            'explanation': explanation,
            'correct_answers': correct_answers,
            'top_docs_sample': top_docs_sample
        }

    def _get_correct_answers(self, relevant_doc_ids):
        """Extract correct answers from relevant documents"""
        correct_answers = []
        for doc_id in relevant_doc_ids:
            if doc_id in self.corpus:
                doc_text = self.corpus[doc_id]
                # Try to extract the answer part (after the query)
                if '?' in doc_text:
                    answer_part = doc_text.split('?', 1)[-1].strip()
                    if answer_part:
                        correct_answers.append(answer_part[:200] + "..." if len(answer_part) > 200 else answer_part)
                else:
                    correct_answers.append(doc_text[:200] + "..." if len(doc_text) > 200 else doc_text)

        return correct_answers if correct_answers else ["Answer not found in corpus"]

    def _get_top_docs_sample(self, top_k_doc_ids, max_samples=3):
        """Get sample of top retrieved documents with their content"""
        samples = []
        for doc_id in top_k_doc_ids[:max_samples]:
            if doc_id in self.corpus:
                doc_text = self.corpus[doc_id]
                samples.append((doc_id, doc_text[:100] + "..." if len(doc_text) > 100 else doc_text))
            else:
                samples.append((doc_id, "Document not found in corpus"))
        return samples


def load_tsv_data(corpus_file, queries_file, qrels_file, max_queries=None, max_corpus_docs=None):
    """
    Load TSV files with limits and handle different formats
    """
    corpus = {}
    queries = {}
    qrels = defaultdict(dict)

    # Load corpus (format: doc_id\ttext)
    print(f"Loading corpus from {corpus_file}...")
    try:
        with open(corpus_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_corpus_docs and len(corpus) >= max_corpus_docs:
                    break
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        doc_id = parts[0].strip()
                        text = '\t'.join(parts[1:]).strip()  # Handle cases where text contains tabs
                        if doc_id and text:
                            corpus[doc_id] = text
                    else:
                        print(f"Warning: Skipping malformed line {line_num} in corpus: {line.strip()}")
                except Exception as e:
                    print(f"Error processing line {line_num} in corpus: {e}")
    except Exception as e:
        print(f"Error loading corpus file: {e}")

    # Load queries (format: query_id\tquery_text)
    print(f"Loading queries from {queries_file}...")
    try:
        with open(queries_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if max_queries and len(queries) >= max_queries:
                    break
                try:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        query_id = parts[0].strip()
                        query_text = '\t'.join(parts[1:]).strip()
                        if query_id and query_text:
                            queries[query_id] = query_text
                    else:
                        print(f"Warning: Skipping malformed line {line_num} in queries: {line.strip()}")
                except Exception as e:
                    print(f"Error processing line {line_num} in queries: {e}")
    except Exception as e:
        print(f"Error loading queries file: {e}")

    # Load qrels (format: query_id\tdoc_id\trelevance)
    print(f"Loading qrels from {qrels_file}...")
    try:
        with open(qrels_file, 'r', encoding='utf-8') as f:
            # Skip header if exists
            first_line = f.readline().strip()

            # Check if first line is header
            if 'rel' in first_line.lower() or 'relevance' in first_line.lower():
                print("Detected header in qrels file, skipping first line")
            else:
                # Process first line
                f.seek(0)

            for line_num, line in enumerate(f, 1):
                try:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        query_id = parts[0].strip()
                        doc_id = parts[1].strip()
                        # Handle non-integer relevance scores (like 'rel' header)
                        try:
                            relevance = int(parts[2])
                            if relevance > 0:  # Only consider positive relevance
                                qrels[query_id][doc_id] = relevance
                        except ValueError:
                            # Skip lines with non-integer relevance (like headers)
                            if line_num == 1:  # Likely header
                                continue
                            else:
                                print(f"Warning: Non-integer relevance score in line {line_num}: {parts[2]}")
                    elif len(parts) == 2:
                        # Assume binary relevance if only 2 columns
                        query_id = parts[0].strip()
                        doc_id = parts[1].strip()
                        qrels[query_id][doc_id] = 1
                    else:
                        if line_num > 1:  # Don't warn for empty lines
                            print(f"Warning: Skipping malformed line {line_num} in qrels: {line.strip()}")
                except Exception as e:
                    print(f"Error processing line {line_num} in qrels: {e}")
    except Exception as e:
        print(f"Error loading qrels file: {e}")

    print(f"Corpus: {len(corpus)} documents")
    print(f"Queries: {len(queries)} queries")
    print(f"Qrels: {sum(len(docs) for docs in qrels.values())} relevance judgments")

    return dict(corpus), dict(queries), dict(qrels)


def verify_data_consistency(queries, corpus, qrels):
    """Verify that data is loaded correctly with exact format"""
    print("\n" + "=" * 50)
    print("DATA CONSISTENCY CHECK")
    print("=" * 50)

    # Basic counts
    print(f"Total queries: {len(queries)}")
    print(f"Total corpus documents: {len(corpus)}")
    print(f"Total queries with relevance judgments: {len(qrels)}")

    # Check overlap between queries and qrels
    overlapping_queries = set(queries.keys()) & set(qrels.keys())
    print(f"Queries with both text and relevance judgments: {len(overlapping_queries)}")

    # Verify relevant documents exist in corpus
    all_relevant_docs = set()
    for docs_dict in qrels.values():
        all_relevant_docs.update(docs_dict.keys())  # Get keys from the inner dict

    missing_docs = all_relevant_docs - set(corpus.keys())
    print(f"Relevant documents missing from corpus: {len(missing_docs)}")

    if missing_docs:
        print(f"First 3 missing docs: {list(missing_docs)[:3]}")

    # Calculate coverage statistics
    total_relevant_pairs = sum(len(docs) for docs in qrels.values())
    available_relevant_pairs = 0

    for qid, docs_dict in qrels.items():
        if qid in queries:  # Query exists
            # docs_dict is a dictionary, we need to check its keys against corpus
            available_docs = set(docs_dict.keys()) & set(corpus.keys())  # Docs that exist in corpus
            available_relevant_pairs += len(available_docs)

    coverage = available_relevant_pairs / total_relevant_pairs if total_relevant_pairs > 0 else 0
    print(f"Relevance judgment coverage: {available_relevant_pairs}/{total_relevant_pairs} ({coverage:.1%})")

    # Check if we have enough data for meaningful evaluation
    if len(overlapping_queries) == 0:
        print("❌ CRITICAL: No overlapping queries between queries and qrels!")
        return False

    if len(overlapping_queries) < 10:
        print(f"⚠️ WARNING: Only {len(overlapping_queries)} queries available for evaluation")
        print("Proceeding anyway...")
        return True

    print("✅ Good data coverage for evaluation")
    return True


def save_detailed_results_to_excel(detailed_results, metrics, output_file, model_name="DPR"):
    """Save detailed results to Excel file"""
    if not detailed_results:
        print("❌ No detailed results to save")
        return

    # Convert to DataFrame
    df_data = []
    for result in detailed_results:
        row = {
            'query_id': result['query_id'],
            'query_text': result['query_text'],
            'relevant_doc_ids': str(result['relevant_doc_ids']),
            'top_k_doc_ids': str(result['top_k_doc_ids']),
            'top_k_scores': str([f"{score:.4f}" for score in result['top_k_scores']]),
            'first_relevant_rank': result['first_relevant_rank'] if result['first_relevant_rank'] else "Not found",
            'MRR': result['MRR'],
            'Recall': result['Recall'],
            'correct_answers': ' | '.join(result['correct_answers']),
            'explanation': result['explanation'],
            'top_docs_sample': str(result['top_docs_sample'])
        }
        df_data.append(row)

    df = pd.DataFrame(df_data)

    # Save to Excel
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Detailed results sheet
            df.to_excel(writer, sheet_name='Detailed_Results', index=False)

            # Summary sheet
            summary_data = {
                'Metric': [
                    'Model', 'Total Queries', 'Average MRR@10', 'Average Recall@10',
                    'Evaluation Timestamp', 'Queries with relevant docs'
                ],
                'Value': [
                    model_name, len(df), metrics['MRR@10'], metrics['Recall@10'],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), metrics['num_queries']
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Statistics sheet
            stats_data = {
                'Statistic': [
                    'Min MRR', 'Max MRR', 'Mean MRR', 'Std MRR',
                    'Min Recall', 'Max Recall', 'Mean Recall', 'Std Recall',
                    'Queries with MRR > 0', 'Queries with Recall > 0'
                ],
                'Value': [
                    df['MRR'].min(), df['MRR'].max(), df['MRR'].mean(), df['MRR'].std(),
                    df['Recall'].min(), df['Recall'].max(), df['Recall'].mean(), df['Recall'].std(),
                    len(df[df['MRR'] > 0]), len(df[df['Recall'] > 0])
                ]
            }
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)

        print(f"✅ Detailed results saved to: {output_file}")

        # Print first few rows for verification
        print("\n📊 First 3 rows of detailed results:")
        print(df.head(3).to_string(index=False))

    except Exception as e:
        print(f"❌ Failed to save Excel file: {e}")
        # Fallback to CSV
        csv_file = output_file.replace('.xlsx', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"✅ Results saved to CSV as fallback: {csv_file}")


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Configuration
    set_seed(42)

    # File paths
    corpus_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/corpus.tsv"
    queries_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/queries.tsv"
    qrels_file = "/root/pycharm_semanticsearch/dataset/web_questions/test/qrels.tsv"

    # Output files
    OUTPUTS_DIR = "/root/pycharm_semanticsearch"
    RESULTS_FILE = os.path.join(OUTPUTS_DIR, "dpr_evaluation_results.csv")
    DETAILED_RESULTS_FILE = os.path.join(OUTPUTS_DIR, "dpr_detailed_evaluation_results.xlsx")

    # Evaluation settings
    MAX_QUERIES = 3000  # process all queries if None
    MAX_CORPUS_DOCS = 10000000  # process all corpus documents if None
    RECALL_K = 10

    try:
        # Load data with limits
        print("Loading data...")
        corpus, queries, qrels = load_tsv_data(
            corpus_file, queries_file, qrels_file,
            max_queries=MAX_QUERIES,
            max_corpus_docs=MAX_CORPUS_DOCS
        )

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

        # Verify data consistency
        if not verify_data_consistency(queries, corpus, qrels):
            print("Data consistency check failed!")
            return

        # Check for queries that have relevant documents
        queries_with_relevant = [qid for qid in queries if qid in qrels and qrels[qid]]
        print(f"Queries with relevant documents: {len(queries_with_relevant)}")

        if not queries_with_relevant:
            print("No queries with relevant documents found!")
            return

        # Initialize DPR evaluator
        print("Initializing DPR...")
        evaluator = DPREvaluator(corpus, queries, qrels)

        # Evaluate with detailed results
        print("Evaluating DPR...")
        start_time = time.time()

        metrics, detailed_results = evaluator.evaluate(k=RECALL_K, max_queries=MAX_QUERIES)

        end_time = time.time()

        # Print results
        print("\n" + "=" * 50)
        print("DPR EVALUATION RESULTS")
        print("=" * 50)
        for metric, score in metrics.items():
            if metric != "num_queries":
                print(f"{metric}: {score:.4f}")
            else:
                print(f"{metric}: {score}")
        print(f"Evaluation time: {end_time - start_time:.2f} seconds")
        print("=" * 50)

        # Save detailed results to Excel
        save_detailed_results_to_excel(
            detailed_results,
            metrics,
            DETAILED_RESULTS_FILE,
            model_name="DPR (facebook/dpr-question_encoder-single-nq-base)"
        )

        # Save summary to CSV
        with open(RESULTS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['model', 'mrr@10', 'recall@10', 'num_queries', 'timestamp'])
            writer.writerow([
                'DPR',
                metrics['MRR@10'],
                metrics['Recall@10'],
                metrics['num_queries'],
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

        print(f"✅ Summary results saved to: {RESULTS_FILE}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please update the file paths in the main() function")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()