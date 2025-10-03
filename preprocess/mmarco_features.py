import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.common_utils import load_pickle, write_pickle
from torch.utils.data import TensorDataset, DataLoader


def mmarco_template_context(query, passage):
    return (
        f"#Q describes a user question. #A describes a web passage. "
        f"Are they related such that #A correctly answers #Q? Answer Can or Cannot.\n"
        f"#Q: {query}\n#A: {passage}\nAnswer:"
    )


def generate_features_and_inbatch(
    model_dir, neg_pkl_file, task_type, bs, teacher_max_seq_length, num_shards, id_shard
):
    bm25_dict = load_pickle(neg_pkl_file)
    all_sample_list = []
    len_dict = {}
    query_ids_for_each_sample = []

    total_keys = list(bm25_dict.keys())
    shard_size = len(total_keys) / num_shards

    for i, query in enumerate(total_keys):
        if shard_size * id_shard <= i < shard_size * (id_shard + 1):
            doc_list = [x[0] if isinstance(x, tuple) else x for x in bm25_dict[query]]
            len_dict[i] = len(doc_list)
            if task_type == "context":
                qry_doc_list = [mmarco_template_context(query, d) for d in doc_list]
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")
            all_sample_list.extend(qry_doc_list)
            query_ids_for_each_sample.extend([query] * len(doc_list))

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True).cuda()
    model.eval()

    all_inputs = tokenizer(
        text=all_sample_list,
        padding="max_length",
        max_length=teacher_max_seq_length,
        truncation=True,
        return_tensors="pt",
    )
    dataset = TensorDataset(all_inputs["input_ids"], all_inputs["attention_mask"])
    dataloader = DataLoader(dataset, batch_size=bs, pin_memory=True)

    all_logits = []
    all_feature_batches = []

    yes_id = tokenizer.encode("Can")[0]
    no_id = tokenizer.encode("Cannot")[0]

    inbatch_dict = {}
    sample_index = 0

    with torch.no_grad():
        for step, batch in enumerate(tqdm(dataloader, desc="Generating logits and features")):
            batch_input_ids, batch_attention_mask = [t.cuda(non_blocking=True) for t in batch]
            with torch.amp.autocast(device_type="cuda"):
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

            batch_logits = logits[:, -1, [yes_id, no_id]].cpu()

            inbatch_logits = []
            for i in range(batch_logits.size(0)):
                others = [j for j in range(batch_logits.size(0)) if j != i]
                sample_logits = batch_logits[others]
                inbatch_logits.append(sample_logits)

            inbatch_logits_tensor = torch.stack(inbatch_logits)

            if f"global_rank{id_shard}" not in inbatch_dict:
                inbatch_dict[f"global_rank{id_shard}"] = []

            while len(inbatch_dict[f"global_rank{id_shard}"]) <= step:
                inbatch_dict[f"global_rank{id_shard}"].append(None)

            inbatch_dict[f"global_rank{id_shard}"][step] = inbatch_logits_tensor

    res_dict = {}
    start = 0
    for i, query in tqdm(enumerate(total_keys), total=len(total_keys), desc="Rebuilding results"):
        if shard_size * id_shard <= i < shard_size * (id_shard + 1):
            end = start + len_dict[i]
            doc_list = bm25_dict[query]
            logits_list = all_logits[start:end]
            doc_only_list = [x[0] if isinstance(x, tuple) else x for x in doc_list]
            res_dict[query] = list(zip(doc_only_list, logits_list))
            start = end

    return res_dict, all_feature_batches, inbatch_dict


if __name__ == "__main__":
    model_dir = "gpt2"  # Your English LM, optionally GPT-2 tuned for MS MARCO
    hardneg_dir = "../outputs/neg_faiss/mmarco_train_neg.pkl"
    output_logits_pkl_path = "../outputs/logits/mmarco_train_logits.pkl"
    output_features_pkl_path = "../outputs/features/mmarco_train_features.pkl"
    output_inbatch_pkl_path = "../outputs/inbatch/mmarco_train_inbatch.pkl"

    task_type = "context"
    batch_size = 16
    teacher_max_seq_length = 256
    num_shards = 1
    id_shard = 0

    print("Generating logits, features, and in-batch logits for MS MARCO hard negatives...")
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

    write_pickle({f"global_rank{id_shard}": features_list}, output_features_pkl_path)
    print(f"Saved features to {output_features_pkl_path}")

    write_pickle(inbatch_dict, output_inbatch_pkl_path)
    print(f"Saved in-batch similarities to {output_inbatch_pkl_path}")
