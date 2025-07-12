import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.common_utils import load_pickle, write_pickle
from torch.utils.data import TensorDataset, DataLoader

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
            doc_list = [x[0] if isinstance(x, tuple) else x for x in bm_25_dict[query]]
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

    all_inputs = tokenizer(
        text=all_sample_list,
        padding='max_length',
        max_length=teacher_max_seq_length,
        truncation=True,
        return_tensors='pt'
    )
    input_ids = all_inputs['input_ids']
    attention_mask = all_inputs['attention_mask']
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=bs, pin_memory=True)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating logits"):
            batch_input_ids, batch_attention_mask = [t.cuda(non_blocking=True) for t in batch]
            with torch.cuda.amp.autocast():
                logits = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask).logits
            final_logits = logits[:, -1, [yes_id, no_id]].cpu().float().numpy().tolist()
            all_logits.extend(final_logits)

    assert len(all_logits) == len(all_sample_list)

    start = 0
    for i, query in tqdm(enumerate(total_keys), total=len(total_keys), desc="Rebuilding results"):
        if shard_size * id_shard <= i < shard_size * (id_shard + 1):
            end = start + len_dict[i]
            doc_list = bm_25_dict[query]
            logits_list = all_logits[start:end]
            assert len(doc_list) == len(logits_list)
            doc_only_list = [x[0] if isinstance(x, tuple) else x for x in doc_list]
            res_doc_logits = list(zip(doc_only_list, logits_list))  # [(doc, [yes_logit, no_logit])]
            res_dict[query] = res_doc_logits
            start = end

    return res_dict


if __name__ == '__main__':
    model_dir = "gpt2"
    hardneg_dir = "../outputs/neg_faiss/snli_train_neg.pkl"
    output_pkl_path = "../outputs/logits/snli_train_logits.pkl"

    task_type = "sts"
    batch_size = 128
    teacher_max_seq_length = 256
    num_shards = 1
    id_shard = 0

    print("Generating logits for SNLI hard negatives...")
    generated_logits = generate_logits(
        model_dir=model_dir,
        neg_pkl_file=hardneg_dir,
        task_type=task_type,
        bs=batch_size,
        teacher_max_seq_length=teacher_max_seq_length,
        num_shards=num_shards,
        id_shard=id_shard
    )

    write_pickle(generated_logits, output_pkl_path)
    print(f"Saved logits to {output_pkl_path}")
