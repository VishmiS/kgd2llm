import os
import torch
import time
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.common_utils import load_pickle, write_pickle


DEBUG = False  # Set to True to quickly test with a small subset

def context_template_v5(text1, text2):
    return f'#Q describes a question, and #A describes a web passage; they may not be related. Based only on these descriptions and your understanding of the world, decide whether #A can correctly answer the question posed in #Q. Please answer "can" or "cannot".\n#Q: {text1}\n#A: {text2}\nAnswer:'


def generate_logits(model_dir, neg_pkl_file, task_type, batch_size, teacher_max_seq_length, num_shards, id_shard):
    start_time = time.time()

    print("Loading negatives...")
    bm_25_dict = load_pickle(neg_pkl_file)
    total_keys = list(bm_25_dict.keys())

    if DEBUG:
        total_keys = total_keys[:10]  # Load only 10 queries for quick testing

    shard_size = len(total_keys) / num_shards
    all_sample_list = []
    len_dict = {}

    for i, query in enumerate(total_keys):
        if shard_size * id_shard <= i < shard_size * (id_shard + 1):
            doc_list = bm_25_dict[query]
            len_dict[i] = len(doc_list)
            if task_type == 'context':
                qry_doc_list = [context_template_v5(query, d) for d in doc_list]
            else:
                raise ValueError(f'Unsupported task_type: {task_type}')
            all_sample_list.extend(qry_doc_list)

    print(f"Tokenizing {len(all_sample_list)} samples...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, pad_token='<|endoftext|>',
        truncation_side='right', padding_side='left', cache_dir='./models'
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, cache_dir='./models').to('cuda')
    model.eval()

    if task_type == 'context':
        yes_id = tokenizer.encode('can')[0]
        no_id = tokenizer.encode(' cannot')[0]
    else:
        raise ValueError(f'Error: No Task Type {task_type}')

    all_logits = []
    inbatch_samples = all_sample_list.copy()

    print("Starting inference...")
    with torch.no_grad():
        for batch_texts in trange(0, len(all_sample_list), batch_size):
            batch = all_sample_list[batch_texts: batch_texts + batch_size]
            inputs = tokenizer(
                batch,
                padding='max_length',
                max_length=teacher_max_seq_length,
                truncation=True,
                return_tensors='pt'
            )

            input_ids = inputs['input_ids'].to('cuda')
            attention_mask = inputs['attention_mask'].to('cuda')

            with torch.cuda.amp.autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            final_logits = logits[:, -1, [yes_id, no_id]].cpu().float().numpy().tolist()
            all_logits.extend(final_logits)

    assert len(all_logits) == len(all_sample_list)

    print("Packing results...")
    res_dict = {}
    start = 0
    for i, query in enumerate(total_keys):
        if shard_size * id_shard <= i < shard_size * (id_shard + 1):
            end = start + len_dict[i]
            doc_list = bm_25_dict[query]
            logits_list = all_logits[start:end]
            assert len(doc_list) == len(logits_list)
            res_doc_logits = list(zip(doc_list, logits_list))
            res_dict[query] = res_doc_logits
            start = end

    print(f"Total time: {time.time() - start_time:.2f}s")
    return res_dict, inbatch_samples


if __name__ == '__main__':
    task_type = 'context'
    model_dir = 'gpt2'
    neg_file = '../outputs/neg_bi/train/query_hard_negatives.pkl'
    inbatch_out_file = '../outputs/logits/context_train_inbatch.pkl'
    logits_out_file = '../outputs/logits/context_train_features.pkl'

    batch_size = 128  # Reduced for speed & memory
    teacher_max_seq_length = 500
    num_shards = 4
    id_shard = 0

    print(f"\nProcessing {task_type} - train split")
    generated_logits, inbatch_samples = generate_logits(
        model_dir=model_dir,
        neg_pkl_file=neg_file,
        task_type=task_type,
        batch_size=batch_size,
        teacher_max_seq_length=teacher_max_seq_length,
        num_shards=num_shards,
        id_shard=id_shard
    )

    write_pickle(inbatch_samples, inbatch_out_file)
    print(f"Saved inbatch inputs to {inbatch_out_file}")

    write_pickle(generated_logits, logits_out_file)
    print(f"Saved logits/features to {logits_out_file}")
