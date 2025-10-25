# conda activate faiss-gpu-py38
# python /root/pycharm_semanticsearch/preprocess/save_hardneg_faiss2.py

import sys
import pathlib

project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import os
import sys
import csv
import torch
import pathlib
import warnings
import numpy as np
from enum import Enum
from typing import List, Union
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm, trange
from utils.common_utils import set_seed, write_pickle, load_pickle
import faiss
import pynvml

warnings.filterwarnings('ignore')

# Increase CSV field size limit
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

# Set your config here
config = {
    'sets': ['train', 'val'],
    'base_dir': 'dataset/ms_marco',
    'output_base': 'outputs/neg_faiss',
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'K': 20,
    'max_seq_len': 250,
    'use_sample': False,
    'sample_limit': 10000
}


def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


class EncoderType(Enum):
    CLS = 2
    MEAN = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError(f"Invalid encoder type: {s}")


class BaseBertModel:
    def __init__(self, model_name_or_path, max_seq_length=512, encoder_type='CLS'):
        self.encoder_type = EncoderType.from_string(encoder_type)
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        self.plm_model = AutoModel.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plm_model.to(self.device)

    def get_sentence_embeddings(self, input_ids, attention_mask):
        output = self.plm_model(input_ids=input_ids, attention_mask=attention_mask)
        if self.encoder_type == EncoderType.CLS:
            return output.last_hidden_state[:, 0]
        elif self.encoder_type == EncoderType.MEAN:
            token_embeddings = output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        raise NotImplementedError()

    def batch_to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def encode(self, sentences: Union[str, List[str]], batch_size=128, max_seq_length=None):
        """
        Optimized encode() — faster with larger batches and mixed precision.
        """
        self.plm_model.eval()
        if isinstance(sentences, str):
            sentences = [sentences]
        max_seq_length = max_seq_length or self.max_seq_length

        all_embeddings = []
        use_fp16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 7

        with torch.no_grad():
            for start in trange(0, len(sentences), batch_size, desc="Encoding (fast)", dynamic_ncols=True):
                batch = sentences[start:start + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors='pt'
                )
                inputs = self.batch_to_device(inputs)

                # ✅ Mixed precision for faster encoding
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    embs = self.get_sentence_embeddings(inputs["input_ids"], inputs["attention_mask"])
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)

                all_embeddings.append(embs.cpu())

        return torch.cat(all_embeddings, dim=0).numpy()


def save_corpus(model, tsv_path, output_dir, split):
    ensure_dir(output_dir)
    cache_path = f"/root/pycharm_semanticsearch/cache/{split}_corpus_embeddings.npy"

    corpus_set = set()
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            text = row['text'].strip()
            if text:
                corpus_set.add(text)
            if config.get('use_sample', False) and len(corpus_set) >= config.get('sample_limit', float('inf')):
                break

    corpus = list(corpus_set)
    corpus_id_map = {text: idx for idx, text in enumerate(corpus)}

    # ✅ Use cached embeddings if size matches
    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
        print(f"[DEBUG] Cache file found: {cache_path}")
        print(f"[DEBUG] → Cached embedding shape: {embeddings.shape}")
        print(f"[DEBUG] → Corpus text count: {len(corpus)}")

        if embeddings.shape[0] == len(corpus):
            print(f"✅ Loaded cached corpus embeddings from {cache_path}")
        else:
            print(f"⚠️ Cache size mismatch — re-encoding corpus...")
            diff = len(corpus) - embeddings.shape[0]
            print(f"[DEBUG] → Difference: {diff} entries (Corpus={len(corpus)}, Cache={embeddings.shape[0]})")
            embeddings = model.encode(corpus, batch_size=128, max_seq_length=config['max_seq_len'])
            np.save(cache_path, embeddings)
    else:
        print(f"⚙️ Encoding corpus (no cache found)...")
        embeddings = model.encode(corpus, batch_size=128, max_seq_length=config['max_seq_len'])
        np.save(cache_path, embeddings)

    write_pickle(corpus_id_map, os.path.join(output_dir, 'corpus_id_map.pkl'))
    write_pickle(corpus, os.path.join(output_dir, 'corpus_texts.pkl'))
    write_pickle(embeddings, os.path.join(output_dir, 'corpus_embeddings.pkl'))

    print(f"✅ Corpus saved at {output_dir} ({len(corpus)} entries)")




def save_queries_and_hard_negatives(model, positive_path, corpus_embeddings_path,
                                    corpus_texts_path, K, output_path, split):
    # Force K = 4
    K = 4
    print(f"\n[DEBUG] Loading positive pairs from: {positive_path}")

    queries = []
    pos_dict = defaultdict(set)

    with open(positive_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, desc=f"[{split.upper()}] Reading positives", dynamic_ncols=True):
            q = row.get('sentence1', '').strip()
            a = row.get('sentence2', '').strip()
            if q and a:
                queries.append(q)
                pos_dict[q].add(a)
                if config.get('use_sample', False) and len(queries) >= config.get('sample_limit', float('inf')):
                    break

    print(f"[DEBUG] Total queries loaded: {len(queries)}")

    corpus_texts = load_pickle(corpus_texts_path)
    corpus_embeddings = load_pickle(corpus_embeddings_path)
    print(f"[DEBUG] Loaded corpus: {len(corpus_texts)} texts, embeddings {corpus_embeddings.shape}")

    cache_path = f"/root/pycharm_semanticsearch/cache/{split}_query_embeddings.npy"
    if os.path.exists(cache_path):
        print(f"[DEBUG] Loading cached query embeddings from {cache_path}")
        query_embeddings = np.load(cache_path)
    else:
        print(f"[DEBUG] Cache not found — encoding queries (batch_size=256, max_seq_len=100)")
        from time import time
        t0 = time()
        query_embeddings = model.encode(queries, batch_size=256, max_seq_length=100)
        print(f"[DEBUG] Query embeddings computed in {time()-t0:.2f}s (shape={query_embeddings.shape})")
        np.save(cache_path, query_embeddings)

    corpus_embs = corpus_embeddings.astype('float32')
    query_embs = query_embeddings.astype('float32')

    print(f"[DEBUG] Building CPU FAISS index (dim={corpus_embs.shape[1]})...")

    # ✅ Use CPU-only FAISS index
    cpu_index = faiss.IndexFlatIP(corpus_embs.shape[1])

    # ✅ Add corpus embeddings directly on CPU (no batching needed)
    cpu_index.add(corpus_embs)
    print(f"[DEBUG] CPU FAISS index built with {cpu_index.ntotal} corpus entries.")

    # ✅ Search in batches (handles large query sets safely)
    search_k = K + 50
    print(f"[DEBUG] Searching CPU FAISS index (search_k={search_k})...")
    D_list, I_list = [], []
    query_batch = 10_000  # slightly smaller to reduce RAM pressure

    for start in trange(0, query_embs.shape[0], query_batch, desc="[CPU FAISS] Searching", dynamic_ncols=True):
        end = start + query_batch
        D, I = cpu_index.search(query_embs[start:end], search_k)
        D_list.append(D)
        I_list.append(I)

    D = np.vstack(D_list)
    I = np.vstack(I_list)

    print(f"[DEBUG] ✅ FAISS GPU search completed for {query_embs.shape[0]} queries.")

    result = {}
    skipped_low_score, skipped_duplicates = 0, 0

    print(f"[DEBUG] Selecting hard negatives for {len(queries)} queries...")
    for i, (neighbors, scores) in enumerate(
        tqdm(zip(I, D), total=len(I), desc=f"[{split.upper()}] Generating negatives", dynamic_ncols=True)
    ):
        q = queries[i]
        seen = set()
        hard_negs = []
        for idx, score in zip(neighbors, scores):
            candidate = corpus_texts[idx]
            if not candidate or candidate == q or candidate in pos_dict[q] or candidate in seen:
                skipped_duplicates += 1
                continue
            if score < 0.1:
                skipped_low_score += 1
                continue
            seen.add(candidate)
            hard_negs.append(candidate)
            if len(hard_negs) >= K:
                break
        result[q] = hard_negs

    print(f"[DEBUG] Skipped duplicates: {skipped_duplicates}, low-score: {skipped_low_score}")
    print(f"[DEBUG] Avg. hard negatives per query: {np.mean([len(v) for v in result.values()]):.2f}")

    write_pickle(result, output_path)
    print(f"✅ Saved {len(result)} queries with {K} hard negatives each → {output_path}")



def process_dataset(split: str, model: BaseBertModel):
    base_dir = config['base_dir']
    input_dir = os.path.join(base_dir, split)
    ensure_dir(input_dir)

    corpus_path = os.path.join(input_dir, 'corpus.tsv')
    positive_path = os.path.join(input_dir, 'positives.tsv')

    for f in [corpus_path, positive_path]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Missing expected file: {f}")
        else:
            print(f"Found input file: {f}")

    # Use a temporary directory for intermediate corpus embeddings
    tmp_output = os.path.join("/root/pycharm_semanticsearch/tmp", split)
    ensure_dir(tmp_output)

    # Save corpus temporarily (it will be reused for both splits)
    save_corpus(model, corpus_path, tmp_output, split)

    # Define final output filenames (only 2 total files)
    if split == "train":
        final_output_path = "/root/pycharm_semanticsearch/outputs/neg_faiss/mmarco_train_neg.pkl"
    else:
        final_output_path = "/root/pycharm_semanticsearch/outputs/neg_faiss/mmarco_val_neg.pkl"

    # Generate and save 4 hard negatives per query
    save_queries_and_hard_negatives(
        model=model,
        positive_path=positive_path,
        corpus_embeddings_path=os.path.join(tmp_output, 'corpus_embeddings.pkl'),
        corpus_texts_path=os.path.join(tmp_output, 'corpus_texts.pkl'),
        K=4,
        output_path=final_output_path,
        split=split
    )



def main():
    set_seed(42)
    model = BaseBertModel(config['model_name'], config['max_seq_len'], encoder_type='CLS')

    for split in config['sets']:
        print(f"\n--- Processing '{split}' split ---")
        process_dataset(split, model)


if __name__ == '__main__':
    main()
