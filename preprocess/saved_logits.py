import os
import sys
import torch
import argparse
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.common_utils import load_pickle, write_pickle

def sts_template_en(text1, text2):
    return f"#P describes an event or issue. #H is a sentence that may or may not be related. Based only on this information and your world knowledge, is #H absolutely true about #P? Answer yes or no. If uncertain, answer no.\n#P: {text1}\n#H: {text2}\nAnswer:"

def context_template_en(text1, text2):
    return f"#Q is a question. #A is a paragraph that may or may not answer the question. Based only on this information and your world knowledge, can #A correctly answer #Q? Answer yes or no.\n#Q: {text1}\n#A: {text2}\nAnswer:"

def generate_logits(model_dir, neg_pkl_file, task_type, bs, teacher_max_seq_length, num_shards, id_shard):
    bm_25_dict = load_pickle(neg_pkl_file)
    all_sample_list = []
    len_dict = {}
    all_logits = []
    res_dict = {}

    shard_size = len(bm_25_dict) // num_shards

    for i, query in enumerate(bm_25_dict):
        if i >= shard_size * id_shard and i < shard_size * (id_shard + 1):
            doc_list = bm_25_dict[query]
            len_dict[i] = len(doc_list)

            if task_type == 'context':
                qry_doc_list = [context_template_en(query, d) for d in doc_list]
            elif task_type == 'sts':
                qry_doc_list = [sts_template_en(query, d) for d in doc_list]
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            all_sample_list.extend(qry_doc_list)

    print(f"Loaded {len(all_sample_list)} samples for shard {id_shard}")

    if not model_dir:
        raise ValueError("Error: 'model_dir' argument is empty. Please provide a valid pretrained model name or path.")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, pad_token='<|endoftext|>')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    yes_token = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_token = tokenizer.encode("no", add_special_tokens=False)[0]

    with torch.no_grad():
        for start_index in trange(0, len(all_sample_list), bs):
            batch = all_sample_list[start_index: start_index + bs]
            inputs = tokenizer(batch, padding='max_length', max_length=teacher_max_seq_length,
                               truncation=True, return_tensors='pt').to(device)
            logits = model(**inputs).logits
            logits = logits[:, -1, [yes_token, no_token]].cpu().float().numpy().tolist()
            all_logits.extend(logits)

    assert len(all_logits) == len(all_sample_list)

    start = 0
    for i, query in enumerate(bm_25_dict):
        if i >= shard_size * id_shard and i < shard_size * (id_shard + 1):
            end = start + len_dict[i]
            doc_list = bm_25_dict[query]
            logits_list = all_logits[start:end]
            assert len(doc_list) == len(logits_list)
            res_dict[query] = list(zip(doc_list, logits_list))
            start = end

    return res_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='gpt2', type=str,
        help='Path to the pretrained language model directory (e.g., HuggingFace or local model).'
    )
    parser.add_argument('--hardneg_dir', default='../dataset/hard_negatives/bm25_hard_negatives.pkl', type=str,
        help='Path to the input pickle file containing query-to-negative mappings.'
    )
    parser.add_argument('--output_pkl', default='../dataset/logits/sts_logits.pkl',type=str,
        help='Path where the output pickle with logits will be saved.'
    )
    parser.add_argument('--task_type', default='sts', type=str, choices=['sts', 'context'],
        help='Type of task: "sts" for semantic textual similarity or "context" for QA-style matching (default: "sts").'
    )
    parser.add_argument('--bs', default=140, type=int,
        help='Batch size for model inference (default: 140).')
    parser.add_argument('--K', type=int)
    parser.add_argument(
        '--teacher_max_seq_length', default=512, type=int,
        help='Maximum sequence length for teacher model inputs (default: 512).'
    )
    parser.add_argument('--num_shards', default=8, type=int,
        help='Total number of data shards for distributed processing (default: 8).'
    )
    parser.add_argument('--id_shard', default=0, type=int,
        help='Shard index to process in this run (0-based, default: 0).'
    )
    args = parser.parse_args()

    result = generate_logits(
        args.model_dir,
        args.hardneg_dir,
        args.task_type,
        args.bs,
        args.teacher_max_seq_length,
        args.num_shards,
        args.id_shard
    )
    print(f"Saving logits to {args.output_pkl}")
    write_pickle(result, args.output_pkl)
