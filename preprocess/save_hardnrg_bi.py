import os
import sys
import csv
import torch
import argparse
import warnings
import numpy as np
from enum import Enum
from tqdm import trange
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch.nn.functional as F
from utils import common_utils
# For training utils (skeleton imports, actual usage depends on training code)
# import deepspeed
# from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

warnings.filterwarnings('ignore')

# Increase CSV limit
maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt / 10)

pd.set_option('display.max_colwidth', None)

class EncoderType(Enum):
    FIRST_LAST_AVG = 0
    LAST_AVG = 1
    CLS = 2
    POOLER = 3
    MEAN = 4

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return EncoderType[s]
        except KeyError:
            raise ValueError(f"Unknown encoder type: {s}")

class BaseBertModel:
    def __init__(self, model_name_or_path, max_seq_length=512, encoder_type='CLS'):
        encoder_type = EncoderType.from_string(encoder_type) if isinstance(encoder_type, str) else encoder_type
        self.encoder_type = encoder_type
        self.max_seq_length = max_seq_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.plm_model = AutoModel.from_pretrained(model_name_or_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.plm_model.to(self.device)

    def batch_to_device(self, batch, device):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        return batch

    def get_sentence_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        model_output = self.plm_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)

        if self.encoder_type == EncoderType.CLS:
            return model_output.last_hidden_state[:, 0]
        elif self.encoder_type == EncoderType.MEAN:
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.encoder_type == EncoderType.POOLER:
            # Use pooler_output if available
            if model_output.pooler_output is not None:
                return model_output.pooler_output
            else:
                # fallback to CLS token
                return model_output.last_hidden_state[:, 0]
        elif self.encoder_type == EncoderType.FIRST_LAST_AVG:
            first = model_output.hidden_states[1]
            last = model_output.hidden_states[-1]
            seq_length = first.size(1)
            first_avg = F.avg_pool1d(first.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            last_avg = F.avg_pool1d(last.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            final_encoding = F.avg_pool1d(torch.stack([first_avg, last_avg], dim=1).transpose(1, 2), kernel_size=2).squeeze(-1)
            return final_encoding
        elif self.encoder_type == EncoderType.LAST_AVG:
            sequence_output = model_output.last_hidden_state
            seq_length = sequence_output.size(1)
            final_encoding = F.avg_pool1d(sequence_output.transpose(1, 2), kernel_size=seq_length).squeeze(-1)
            return final_encoding
        else:
            raise NotImplementedError(f"Encoder type {self.encoder_type} not implemented")

    def encode(self, sentences, batch_size=32, show_progress_bar=False,
               convert_to_numpy=True, convert_to_tensor=False, device=None,
               normalize_embeddings=True, max_seq_length=None):
        self.plm_model.eval()
        device = device or self.device
        max_seq_length = max_seq_length or self.max_seq_length

        if not sentences:
            print("⚠ No sentences provided to encode(). Returning empty list.")
            return np.empty((0, self.plm_model.config.hidden_size)) if convert_to_numpy else []

        if isinstance(sentences, str):
            sentences = [sentences]
            input_is_string = True
        else:
            input_is_string = False

        all_embeddings = []
        sentences = [str(s) for s in sentences]
        length_sorted_idx = np.argsort([-len(s) for s in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            with torch.no_grad():
                features = self.tokenizer(sentences_batch, max_length=max_seq_length, padding=True, truncation=True, return_tensors='pt')
                features = self.batch_to_device(features, device)
                embeddings = self.get_sentence_embeddings(**features)
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                if convert_to_numpy:
                    embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]
        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = torch.stack(all_embeddings).numpy()

        if input_is_string:
            return all_embeddings[0]
        return all_embeddings


def encode_corpus(model, df, output_dir):
    common_utils.makedirs(output_dir)
    print("Encoding corpus...")

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    doc_texts = (df['title'].fillna('') + ' ' + df['body'].fillna('')).astype(str).tolist()
    doc_ids = df.index.astype(str).tolist()

    docid_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}
    idx_to_docid = {i: doc_id for doc_id, i in docid_to_idx.items()}

    embeddings = model.encode(doc_texts, batch_size=64, show_progress_bar=True, convert_to_tensor=True)

    common_utils.write_pickle(docid_to_idx, os.path.join(output_dir, 'corpus_docid_to_idx.pkl'))
    common_utils.write_pickle(idx_to_docid, os.path.join(output_dir, 'corpus_idx_to_docid.pkl'))
    common_utils.write_pickle(embeddings, os.path.join(output_dir, 'corpus_embeddings.pkl'))
    print("Corpus encoding complete.")

def encode_queries(model, queries, output_dir, batch_size=64):
    common_utils.makedirs(output_dir)
    print("Encoding queries...")

    queries = [str(q).strip() for q in queries]
    query_embeddings = model.encode(queries, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True)

    common_utils.write_pickle(queries, os.path.join(output_dir, 'queries.pkl'))
    common_utils.write_pickle(query_embeddings, os.path.join(output_dir, 'query_embeddings.pkl'))
    print("Query encoding complete.")

def mine_hard_negatives(model, corpus_embeddings, corpus_idx_to_docid, queries, query_embeddings, K, output_dir):
    print("Mining hard negatives...")

    res = defaultdict(list)
    corpus_embeddings = corpus_embeddings.to(model.device)
    query_embeddings = query_embeddings.to(model.device)

    qry_num = query_embeddings.size(0)
    batch_size = 2000

    for start in trange(0, qry_num, batch_size, desc="Hard Negative Mining"):
        end = min(start + batch_size, qry_num)
        batch_query_embeds = query_embeddings[start:end]
        scores = common_utils.cos_sim(batch_query_embeds, corpus_embeddings)  # [batch_size, corpus_size]
        topk_scores, topk_indices = torch.topk(scores, K + 1, dim=1, largest=True, sorted=True)

        for i, query_idx in enumerate(range(start, end)):
            query_text = queries[query_idx]
            retrieved_doc_ids = [corpus_idx_to_docid[idx.item()] for idx in topk_indices[i]]
            # Remove positives or duplicates (if you have a positive mapping, filter here)
            # For now, just skip the first (assumed positive)
            hard_negatives = retrieved_doc_ids[1:K + 1]
            res[query_text].extend(hard_negatives)

    # Save hard negatives
    hard_neg_path = os.path.join(output_dir, 'bi_hard_negatives.pkl')
    common_utils.write_pickle(res, hard_neg_path)
    print(f"Hard negatives saved to {hard_neg_path}")

def main():
    # Define device upfront
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='T2Ranking', type=str,
                        help='Name of the dataset to use, e.g., "T2Ranking"')
    parser.add_argument('--fulldocs_pkl', default='../dataset/bi_marco_pkl/fulldocs.pkl', type=str,
                        help='Path to full document corpus pickle file')
    parser.add_argument('--queries_pkl', default='../dataset/bi_marco_pkl/queries.pkl', type=str,
                        help='Path to queries pickle file')
    parser.add_argument('--output_dir', default='../dataset/hard_negatives', type=str,
                        help='Output directory to store generated hard negatives and embeddings')
    parser.add_argument('--K', default=100, type=int,
                        help='Number of hard negatives to mine per query')
    parser.add_argument('--model_name', default='bert-base-uncased', type=str,
                        help='Pretrained model name or path')
    parser.add_argument('--max_seq_len', default=512, type=int,
                        help='Maximum sequence length for the encoder')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--sample_size', default=-1, type=int,
                        help='Number of queries to process (set -1 to use all)')
    parser.add_argument('--encoder_type', default='CLS', type=str,
                        help='Pooling type: CLS, MEAN, POOLER, FIRST_LAST_AVG, LAST_AVG')

    args = parser.parse_args()

    common_utils.set_seed(args.seed)
    common_utils.makedirs(args.output_dir)

    print("Loading corpus data...")
    fulldocs = common_utils.load_pickle(args.fulldocs_pkl)

    model = BaseBertModel(args.model_name, max_seq_length=args.max_seq_len, encoder_type=args.encoder_type)

    # Placeholder for PEFT and Int8 quantization if you want to finetune:
    # model.plm_model = prepare_model_for_int8_training(model.plm_model)
    # peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    # model.plm_model = get_peft_model(model.plm_model, peft_config)
    # model.plm_model = model.plm_model.to(model.device)

    corpus_dir = os.path.join(args.output_dir, 'corpus')
    common_utils.makedirs(corpus_dir)

    corpus_paths = {
        'docid_to_idx': os.path.join(corpus_dir, 'corpus_docid_to_idx.pkl'),
        'idx_to_docid': os.path.join(corpus_dir, 'corpus_idx_to_docid.pkl'),
        'embeddings': os.path.join(corpus_dir, 'corpus_embeddings.pkl'),
    }

    if not os.path.exists(corpus_paths['embeddings']):
        encode_corpus(model, fulldocs, corpus_dir)

    idx_to_docid = common_utils.load_pickle(corpus_paths['idx_to_docid'])
    corpus_embeddings = common_utils.load_pickle(corpus_paths['embeddings'])

    print("Loading queries...")
    queries = common_utils.load_pickle(args.queries_pkl)

    # Convert queries to list of strings if it's a DataFrame or Series
    if isinstance(queries, pd.DataFrame):
        queries = queries.iloc[:, 0].tolist()
    elif isinstance(queries, pd.Series):
        queries = queries.tolist()

    if args.sample_size > 0:
        queries = queries[:args.sample_size]

    query_dir = os.path.join(args.output_dir, 'queries')
    common_utils.makedirs(query_dir)

    query_embeddings_path = os.path.join(query_dir, 'query_embeddings.pkl')

    if not os.path.exists(query_embeddings_path):
        encode_queries(model, queries, query_dir)

    query_embeddings = common_utils.load_pickle(query_embeddings_path)

    # Mine hard negatives
    mine_hard_negatives(model, corpus_embeddings, idx_to_docid, queries, query_embeddings, args.K, args.output_dir)

    print("Process complete.")

if __name__ == '__main__':
    main()
