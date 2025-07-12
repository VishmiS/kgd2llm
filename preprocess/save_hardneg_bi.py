# conda activate faiss-gpu-py38
# python -m preprocess.save_hardneg_bi

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
from utils.common_utils import set_seed, write_pickle, load_pickle, cos_sim
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
import faiss

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
    'sets': ['train', 'dev'],
    'base_dir': 'dataset/ms_marco',  # directory containing train/test tsv files
    'output_base': 'outputs/neg_bi',
    'model_name': 'bert-base-uncased',
    'K': 100,
    'max_seq_len': 250,
    'use_sample': False,           # Toggle this to enable/disable sampling
    'sample_limit': 10000           # How many samples to use when sampling
}

def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def makedirs(path):
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


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

    def encode(self, sentences: Union[str, List[str]], batch_size=32, max_seq_length=None):
        self.plm_model.eval()
        if isinstance(sentences, str):
            sentences = [sentences]
        max_seq_length = max_seq_length or self.max_seq_length
        embeddings = []

        for start in trange(0, len(sentences), batch_size, desc="Encoding", leave=False, dynamic_ncols=True):
            batch = sentences[start:start + batch_size]
            with torch.no_grad():
                inputs = self.tokenizer(batch, padding=True, truncation=True,
                                        max_length=max_seq_length, return_tensors='pt')
                inputs = self.batch_to_device(inputs)
                with autocast():
                    embs = self.get_sentence_embeddings(inputs["input_ids"], inputs["attention_mask"])
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                embeddings.extend(embs.cpu().numpy())
        return np.array(embeddings)


def save_corpus(model, tsv_path, output_dir):
    makedirs(output_dir)
    corpus_set = set()
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            text = row['text'].strip()
            if text:
                corpus_set.add(text[:320])
            if config.get('use_sample', False) and len(corpus_set) >= config.get('sample_limit', float('inf')):
                break

    corpus = list(corpus_set)
    corpus_id_map = {text: idx for idx, text in enumerate(corpus)}

    embeddings = model.encode(corpus, batch_size=512, max_seq_length=250)
    write_pickle(corpus_id_map, os.path.join(output_dir, 'corpus_id_map.pkl'))
    print(f"Saved corpus_id_map.pkl at {output_dir}")
    write_pickle(corpus, os.path.join(output_dir, 'corpus_texts.pkl'))
    print(f"Saved corpus_texts.pkl at {output_dir}")
    write_pickle(embeddings, os.path.join(output_dir, 'corpus_embeddings.pkl'))
    print(f"Saved corpus_embeddings.pkl at {output_dir}")

def save_queries_and_hard_negatives(model, query_path, positive_path, corpus_embeddings_path,
                                    corpus_texts_path, K, output_path):
    queries = []
    pos_dict = defaultdict(list)

    with open(positive_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            q = row.get('sentence1')
            a = row.get('sentence2')
            if q is None or a is None:
                # Skip rows where these fields are missing or None
                continue
            q = q.strip()
            a = a.strip()
            if not q or not a:
                # Skip if empty after stripping
                continue
            queries.append(q)
            pos_dict[q].append(a)
            if config.get('use_sample', False) and len(queries) >= config.get('sample_limit', float('inf')):
                break

    print(f"Number of queries read: {len(queries)}")
    print(f"Number of queries with positives: {len(pos_dict)}")

    corpus_texts = load_pickle(corpus_texts_path)
    corpus_embeddings = load_pickle(corpus_embeddings_path)
    query_embeddings = model.encode(queries, batch_size=256, max_seq_length=100)
    write_pickle(query_embeddings, os.path.join(os.path.dirname(output_path), 'query_embeddings.pkl'))
    print(f"Saved query_embeddings.pkl at {os.path.dirname(output_path)}")

    corpus_embs = corpus_embeddings.astype('float32')
    query_embs = query_embeddings.astype('float32')

    index = faiss.IndexFlatIP(corpus_embs.shape[1])
    index.add(corpus_embs)

    D, I = index.search(query_embs, K + 10)

    result = defaultdict(list)
    for i, neighbors in enumerate(I):
        q = queries[i]
        for idx in neighbors:
            candidate = corpus_texts[idx]
            if candidate not in pos_dict[q] and candidate not in result[q]:
                result[q].append(candidate)
            if len(result[q]) >= K:
                break

    write_pickle(result, output_path)
    print(f"Saved query_hard_negatives.pkl at {output_path}")

def process_dataset(split: str, model: BaseBertModel):
    base_dir = config['base_dir']
    output_dir = os.path.join(config['output_base'], split)

    # Create input directory if it doesn't exist
    input_dir = os.path.join(base_dir, split)
    ensure_dir(input_dir)  # This creates the directory if missing

    # Create output directory
    ensure_dir(output_dir)

    corpus_path = os.path.join(input_dir, 'corpus.tsv')
    query_path = os.path.join(input_dir, 'queries.tsv')
    positive_path = os.path.join(input_dir, 'positives.tsv')

    # You might want to check if these input files exist before continuing:
    for f in [corpus_path, query_path, positive_path]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f"Expected file not found: {f}")
        else:
            print(f"Found input file: {f}")

    save_corpus(model, corpus_path, output_dir)

    save_queries_and_hard_negatives(
        model=model,
        query_path=query_path,
        positive_path=positive_path,
        corpus_embeddings_path=os.path.join(output_dir, 'corpus_embeddings.pkl'),
        corpus_texts_path=os.path.join(output_dir, 'corpus_texts.pkl'),
        K=config['K'],
        output_path=os.path.join(output_dir, 'query_hard_negatives.pkl')
    )



def main():
    set_seed(42)
    model = BaseBertModel(config['model_name'], config['max_seq_len'], encoder_type='CLS')

    for split in config['sets']:
        print(f"\n--- Processing {split} split ---")
        process_dataset(split, model)


if __name__ == '__main__':
    main()
