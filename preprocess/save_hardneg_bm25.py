import os
import pickle
from collections import defaultdict
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import argparse

def write_pickle(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

def preprocess_text(text):
    # Simple tokenization for English (split by whitespace and lowercasing)
    if not isinstance(text, str):
        return []
    return text.lower().split()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos_pkl', default='../dataset/bm25_combined_pkl/combined_pos.pkl', type=str,
                        help='Path to positive samples pickle')
    parser.add_argument('--hn_pkl', default='../dataset/bm25_combined_pkl/combined_hn.pkl', type=str,
                        help='Path to hard negative samples pickle')
    parser.add_argument('--output_dir', default='../dataset/hard_negatives', type=str,
                        help='Output directory for hard negatives')
    parser.add_argument('--K', default=10, type=int, help='Number of hard negatives to mine per query')
    parser.add_argument('--sample_size', default=10000, type=int,
                        help='Optional: Use only the first N positive samples for debugging')

    args = parser.parse_args()

    print("Loading positive and hard negative samples...")
    pos_samples = load_pickle(args.pos_pkl)  # list of [sentence1, sentence2]
    hn_samples = load_pickle(args.hn_pkl)    # list of [sentence1, sentence2]

    # Optionally reduce sample size for debugging
    if args.sample_size:
        pos_samples = pos_samples[:args.sample_size]
        hn_samples = hn_samples[:args.sample_size * 2]  # maintain larger pool for mining
        print(f"Using sample_size={args.sample_size}: {len(pos_samples)} positives, {len(hn_samples)} negatives")

    print(f"pos_samples type: {type(pos_samples)}, length: {len(pos_samples)}")
    if len(pos_samples) > 0:
        print(f"First positive sample: {pos_samples[0]}")

    print(f"Loaded {len(pos_samples)} positive samples and {len(hn_samples)} negative samples.")

    # Build corpus from all sentence2s (including positives and negatives)
    corpus_sentences = list(set([sent2 for _, sent2 in pos_samples + hn_samples]))

    # Tokenize corpus (simple whitespace lowercase split)
    tokenized_corpus = [preprocess_text(doc) for doc in corpus_sentences]

    # Initialize BM25 with English tokenized corpus
    bm25 = BM25Okapi(tokenized_corpus)

    # Extract queries (sentence1 from positive samples)
    queries = [sent1 for sent1, _ in pos_samples]
    tokenized_queries = [preprocess_text(q) for q in queries]

    # Build positive mapping: query -> list of its positive sentence2's
    pos_sample_dict = defaultdict(list)
    for sent1, sent2 in pos_samples:
        pos_sample_dict[sent1].append(sent2)

    # Hard negative mining dictionary: query -> list of hard negatives (sentence2s)
    hard_neg_sample_dict = defaultdict(list)

    print("Mining hard negatives using BM25...")
    for i, tokenized_query in enumerate(tqdm(tokenized_queries)):
        # Get top K+10 (buffer) results from BM25 to filter later
        top_docs = bm25.get_top_n(tokenized_query, corpus_sentences, n=args.K + 10)
        # Remove positives from the candidate hard negatives
        filtered_negatives = [doc for doc in top_docs if doc not in pos_sample_dict[queries[i]]]
        # Take top K after filtering
        hard_neg_sample_dict[queries[i]] = filtered_negatives[:args.K]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_pickle = os.path.join(args.output_dir, 'bm25_hard_negatives.pkl')
    write_pickle(hard_neg_sample_dict, output_pickle)

    print(f"Hard negatives saved to {output_pickle}")

if __name__ == '__main__':
    main()
