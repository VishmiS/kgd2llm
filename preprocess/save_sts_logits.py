import os
import torch
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.common_utils import load_pickle, write_pickle


def sts_template_v5(text1, text2):
    return f"#P and #H each describe an event or situation. Based only on this and your knowledge of the world, is #H definitely true if #P is true? Answer yes or no.\n#P: {text1}\n#H: {text2}\nAnswer:"


def generate_logits(model_dir, neg_pkl_file, task_type, bs, teacher_max_seq_length, num_shards, id_shard):
    bm_25_dict = load_pickle(neg_pkl_file)
    all_sample_list = []
    len_dict = {}
    all_logits = []
    res_dict = {}

    total_keys = list(bm_25_dict.keys())
    shard_size = len(total_keys) / num_shards

    for i, query in enumerate(total_keys):
        if shard_size * id_shard <= i < shard_size * (id_shard + 1):
            doc_list = bm_25_dict[query]
            len_dict[i] = len(doc_list)
            if task_type == 'sts':
                qry_doc_list = [sts_template_v5(query, d) for d in doc_list]
            else:
                raise ValueError(f'Unsupported task_type: {task_type}')
            all_sample_list.extend(qry_doc_list)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True, pad_token='<|endoftext|>',
        truncation_side='right', padding_side='left'
    )
    tokenizer.pad_token_id = tokenizer.eod_id if hasattr(tokenizer, 'eod_id') else tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to('cuda')
    model.eval()

    if task_type == 'sts':
        yes_id = tokenizer.encode('yes')[0]
        no_id = tokenizer.encode('no')[0]
    else:
        raise ValueError(f'Error: No Task Type {task_type}')

    # Tokenize entire list once
    all_inputs = tokenizer(
        text=all_sample_list,
        padding='max_length',
        max_length=teacher_max_seq_length,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = all_inputs['input_ids'].to('cuda')
    attention_mask = all_inputs['attention_mask'].to('cuda')

    inbatch_samples = all_sample_list.copy()  # We already have all inputs here

    with torch.no_grad():
        # Use mixed precision for faster inference
        scaler = torch.cuda.amp.GradScaler(enabled=False)  # Just for autocast; no scaling needed
        for start_index in trange(0, len(all_sample_list), bs, disable=False):
            batch_input_ids = input_ids[start_index: start_index + bs]
            batch_attention_mask = attention_mask[start_index: start_index + bs]

            with torch.cuda.amp.autocast():
                logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits

            final_logits = logits[:, -1, [yes_id, no_id]].cpu().float().numpy().tolist()
            all_logits.extend(final_logits)

    assert len(all_logits) == len(all_sample_list)

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

    return res_dict, inbatch_samples


if __name__ == '__main__':
    model_dir = "gpt2"
    hardneg_dir = "../outputs/neg_faiss/sts_train_neg.pkl"

    INBATCH_PKL_PATH_DIR = "../outputs/logits/sts_train_inbatch.pkl"
    FEATURE_PKL_PATH_DIR = "../outputs/logits/sts_train_features.pkl"

    task_type = "sts"
    batch_size = 128
    teacher_max_seq_length = 256
    num_shards = 1
    id_shard = 0

    print("Generating logits for STS hard negatives...")
    generated_logits, inbatch_samples = generate_logits(
        model_dir=model_dir,
        neg_pkl_file=hardneg_dir,
        task_type=task_type,
        bs=batch_size,
        teacher_max_seq_length=teacher_max_seq_length,
        num_shards=num_shards,
        id_shard=id_shard
    )

    write_pickle(inbatch_samples, INBATCH_PKL_PATH_DIR)
    print(f"Saved inbatch inputs to {INBATCH_PKL_PATH_DIR}")

    write_pickle(generated_logits, FEATURE_PKL_PATH_DIR)
    print(f"Saved logits/features to {FEATURE_PKL_PATH_DIR}")
