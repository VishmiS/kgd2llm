import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.common_utils import load_pickle, write_pickle
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


def sts_template_v5(text1, text2):
    return f"#P and #H each describe an event or situation. Based only on this and your knowledge of the world, is #H definitely true if #P is true?\n#P: {text1}\n#H: {text2}\nAnswer:"


def generate_features_and_inbatch(
    model_dir, neg_pkl_file, task_type, bs, teacher_max_seq_length, num_shards, id_shard
):
    bm_25_dict = load_pickle(neg_pkl_file)
    all_sample_list = []
    len_dict = {}
    query_ids_for_each_sample = []

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
            query_ids_for_each_sample.extend([query] * len(doc_list))  # <-- Track query ID for each sample

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        pad_token='<|endoftext|>',
        truncation_side='right',
        padding_side='left',
    )
    tokenizer.pad_token_id = tokenizer.eod_id if hasattr(tokenizer, 'eod_id') else tokenizer.pad_token_id

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).to('cuda')
    model.eval()

    all_inputs = tokenizer(
        text=all_sample_list,
        padding='max_length',
        max_length=teacher_max_seq_length,
        truncation=True,
        return_tensors='pt',
    )
    input_ids = all_inputs['input_ids']
    attention_mask = all_inputs['attention_mask']
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=bs, pin_memory=True)

    all_logits = []
    all_feature_batches = []

    yes_id = tokenizer.encode('yes')[0]
    no_id = tokenizer.encode('no')[0]

    inbatch_dict = {}
    sample_index = 0  # Tracks where we are in `query_ids_for_each_sample`

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Generating logits and features")):
            batch_input_ids, batch_attention_mask = [t.cuda(non_blocking=True) for t in batch]
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                )

            last_hidden_states = outputs.hidden_states[-1]
            mask_expanded = batch_attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_hidden = torch.sum(last_hidden_states * mask_expanded, dim=1)
            lengths = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_features = (sum_hidden / lengths).to(torch.float16).cpu()

            all_feature_batches.append(pooled_features)

            logits = outputs.logits
            final_logits = logits[:, -1, [yes_id, no_id]].cpu().float().numpy().tolist()
            all_logits.extend(final_logits)

            feat_stack = pooled_features
            feat_norm = F.normalize(feat_stack, p=2, dim=1)
            cos_sim_matrix = torch.matmul(feat_norm, feat_norm.T)
            query_ids_batch = query_ids_for_each_sample[sample_index: sample_index + feat_stack.size(0)]

            if f'global_rank{id_shard}' not in inbatch_dict:
                inbatch_dict[f'global_rank{id_shard}'] = []

            # Make sure the list is long enough
            while len(inbatch_dict[f'global_rank{id_shard}']) <= step:
                inbatch_dict[f'global_rank{id_shard}'].append(None)

            inbatch_dict[f'global_rank{id_shard}'][step] = cos_sim_matrix.cpu()

            sample_index += feat_stack.size(0)

    # Build logits dictionary
    res_dict = {}
    start = 0
    for i, query in tqdm(enumerate(total_keys), total=len(total_keys), desc="Rebuilding results"):
        if shard_size * id_shard <= i < shard_size * (id_shard + 1):
            end = start + len_dict[i]
            doc_list = bm_25_dict[query]
            logits_list = all_logits[start:end]
            doc_only_list = [x[0] if isinstance(x, tuple) else x for x in doc_list]
            res_dict[query] = list(zip(doc_only_list, logits_list))
            start = end

    return res_dict, all_feature_batches, inbatch_dict


if __name__ == '__main__':
    model_dir = "gpt2"
    hardneg_dir = "../outputs/neg_faiss/snli_train_neg.pkl"
    output_logits_pkl_path = "../outputs/logits/snli_train_logits.pkl"
    output_features_pkl_path = "../outputs/features/snli_train_features.pkl"
    output_inbatch_pkl_path = "../outputs/inbatch/snli_train_inbatch.pkl"

    task_type = "sts"
    batch_size = 128
    teacher_max_seq_length = 256
    num_shards = 1
    id_shard = 0

    print("Generating logits, features, and inbatch similarities for SNLI hard negatives...")
    generated_logits, features_list, inbatch_dict = generate_features_and_inbatch(
        model_dir=model_dir,
        neg_pkl_file=hardneg_dir,
        task_type=task_type,
        bs=batch_size,
        teacher_max_seq_length=teacher_max_seq_length,
        num_shards=num_shards,
        id_shard=id_shard,
    )

    write_pickle(generated_logits, output_logits_pkl_path)
    print(f"Saved logits to {output_logits_pkl_path}")

    write_pickle({f'global_rank{id_shard}': features_list}, output_features_pkl_path)
    print(f"Saved features to {output_features_pkl_path}")

    write_pickle(inbatch_dict, output_inbatch_pkl_path)
    print(f"Saved inbatch similarities to {output_inbatch_pkl_path}")
