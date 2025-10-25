import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
import csv
from loguru import logger
import random
from utils.common_utils import load_pickle
import pickle

DATASET_ID_DICT = {'snli': 1, 'sts': 2, 'mmarco': 3, 'wq': 4}


def load_text_dataset(name, pos_dir, neg_dir, file_path, neg_K, res_data, split):
    print(f"\n🔄 Loading dataset '{name}' split='{split}'")
    print(f"  ▶️ Positive pickle file: {pos_dir}")
    print(f"  ▶️ Negative pickle file: {neg_dir}")
    print(f"  ▶️ Data file (TSV/JSONL): {file_path}\n")
    missing_hard_neg_count = 0  # Track how many samples are missing hard negatives

    data = []
    if split == 'train':
        hard_neg_house = load_pickle(neg_dir)
        pos_logits = load_pickle(pos_dir)
        if not isinstance(pos_logits, dict):
            raise ValueError(f"[ERROR] Expected pos_logits to be a dict, got {type(pos_logits)} instead.")
        print(f"✅ Loaded pos_logits with {len(pos_logits)} entries from {pos_dir}")

    elif split == 'validation':
        hard_neg_house = {}  # no hard negatives for validation
        pos_logits = load_pickle(pos_dir)  # load validation pos logits
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        print(f"🔑 TSV Header Columns for file '{file_path}':", reader.fieldnames)
        for id, row in enumerate(reader):
            text_a = row['sentence1']
            text_b = row['sentence2']
            score = row['gold_label']
            if score == 'entailment':
                if split == 'train':
                    neg_list = hard_neg_house.get(text_a, [])
                    if not neg_list:
                        missing_hard_neg_count += 1
                        continue

                    if text_a not in hard_neg_house or len(hard_neg_house[text_a]) == 0:
                        continue  # avoid division by zero or missing key

                    if len(hard_neg_house[text_a]) < neg_K:
                        num = math.ceil(neg_K / len(hard_neg_house[text_a]))
                        negs_logits = random.sample(hard_neg_house[text_a] * num, neg_K)
                    else:
                        negs_logits = random.sample(hard_neg_house[text_a], neg_K)
                    hardnegs, hardneg_logits = zip(*negs_logits)
                    hardnegs, hardneg_logits = list(hardnegs), list(hardneg_logits)
                elif split == 'validation':
                    hardnegs = []
                    hardneg_logits = []
                hardnegs = [sample[:100] for sample in hardnegs]
                # data.append((text_a[:100], text_b[:100], pos_logits, hardnegs, hardneg_logits, 0))

                not_found_count = 0  # Initialize this before your loop
                # Inside your loop or wherever this line is executed:
                pos_logit_for_sample = pos_logits.get(text_a, None)
                if pos_logit_for_sample is None:
                    not_found_count += 1
                    pos_logit_for_sample = [0.0, 0.0]  # assign default after counting
                # Then append to data as usual
                data.append((text_a[:100], text_b[:100], pos_logit_for_sample, hardnegs, hardneg_logits, 0))

    if split == 'train' and missing_hard_neg_count > 0:
        print(f"⚠️ Total queries with NO hard negatives: {missing_hard_neg_count}")

    print(f"how many times text_a is not found in pos_logits: {not_found_count}")

    if split == 'train':
        split_data = data[:-10000]
        sample_num = len(split_data)
    elif split == 'validation':
        split_data = data[-10000:]
        sample_num = len(split_data)
    res_data.extend(split_data)

    return res_data, sample_num


def load_sts_dataset_train(name, pos_dir, neg_dir, file_path, neg_K, res_data, split='train'):
    import csv, math, random

    print(f"\n🔄 Loading dataset '{name}' split='{split}'")
    print(f"  ▶️ Positive pickle file: {pos_dir}")
    print(f"  ▶️ Negative pickle file: {neg_dir}")
    print(f"  ▶️ Data file: {file_path}\n")

    data = []
    missing_hard_neg_count = 0
    not_found_count = 0

    # Load pickles depending on split
    if split == 'train':
        hard_neg_house = load_pickle(neg_dir)
        pos_logits = load_pickle(pos_dir)
        if not isinstance(pos_logits, dict):
            raise ValueError(f"[ERROR] Expected pos_logits to be a dict, got {type(pos_logits)} instead.")
        print(f"✅ Loaded pos_logits with {len(pos_logits)} entries from {pos_dir}")

    else:
        hard_neg_house = {}
        pos_logits = load_pickle(pos_dir)
        if not isinstance(pos_logits, dict):
            raise ValueError(f"[ERROR] Expected pos_logits to be a dict, got {type(pos_logits)} instead.")
        print(f"✅ Loaded pos_logits with {len(pos_logits)} entries from {pos_dir}")

    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        print(f"🔑 CSV Header Columns for file '{file_path}':", reader.fieldnames)

        for id, row in enumerate(reader):
            text_a = row['sentence1']
            text_b = row['sentence2']

            # Training-specific logic
            if split == 'train':
                neg_list = hard_neg_house.get(text_a, [])
                if not neg_list:
                    missing_hard_neg_count += 1
                    continue

                if len(neg_list) < neg_K:
                    num = math.ceil(neg_K / len(neg_list))
                    negs_logits = random.sample(neg_list * num, neg_K)
                else:
                    negs_logits = random.sample(neg_list, neg_K)

                hardnegs, hardneg_logits = zip(*negs_logits)
                hardnegs = [sample[:100] for sample in hardnegs]
                hardneg_logits = list(hardneg_logits)
            else:
                hardnegs, hardneg_logits = [], []

            # Get positive logits
            pos_logit_for_sample = pos_logits.get(text_a)
            if pos_logit_for_sample is None:
                not_found_count += 1
                pos_logit_for_sample = [0.0, 0.0]

            # Trim sentences to 100 characters
            data.append((text_a[:100], text_b[:100], pos_logit_for_sample, hardnegs, hardneg_logits, 0))

    # Reporting
    if split == 'train':
        print(f"⚠️ Total queries with NO hard negatives: {missing_hard_neg_count}")
    print(f"📉 Pos logits not found for {not_found_count} samples.")

    # Optional split logic
    if split == 'train':
        split_data = data[:-10000] if len(data) > 10000 else data
    elif split == 'validation':
        split_data = data[-10000:] if len(data) > 10000 else data
    else:
        split_data = data

    sample_num = len(split_data)
    res_data.extend(split_data)

    return res_data, sample_num


def load_sts_dataset_val(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = []
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for id, row in enumerate(reader):
            text_a = row['sentence1']
            text_b = row['sentence2']
            data.append((text_a[:100], text_b[:100], [], [], [], 0))

    sample_num = len(data)
    res_data.extend(data)

    return res_data, sample_num

# ======== FAST LOOKUP REPLACEMENT ========

from collections import defaultdict

# Global caches for fast lookup
_pos_tuple_map_cache = {}
_pos_query_map_cache = {}

def build_fast_lookup_maps(pos_dict):
    """Build efficient lookup tables for O(1) access."""
    if id(pos_dict) in _pos_tuple_map_cache:
        # Already built
        return _pos_tuple_map_cache[id(pos_dict)], _pos_query_map_cache[id(pos_dict)]

    pos_tuple_map = {}
    pos_query_map = defaultdict(dict)

    for k, v in pos_dict.items():
        if isinstance(k, tuple) and len(k) == 2:
            q, a = k
        else:
            # handle serialized keys
            parts = str(k).split("|||")
            if len(parts) == 2:
                q, a = parts
            else:
                q, a = str(k), ""

        qn = q.strip().lower()
        an = a.strip().lower()

        pos_tuple_map[(q, a)] = v
        pos_query_map[qn][an] = v

    _pos_tuple_map_cache[id(pos_dict)] = pos_tuple_map
    _pos_query_map_cache[id(pos_dict)] = pos_query_map
    return pos_tuple_map, pos_query_map


def fast_lookup_pos_logit(pos_dict, query, answer):
    """Ultra-fast lookup (~O(1)) instead of scanning all 500k entries."""
    pos_tuple_map, pos_query_map = build_fast_lookup_maps(pos_dict)
    # print(f"⚡ Built fast lookup maps for {len(pos_tuple_map)} entries "
    #       f"and {len(pos_query_map)} unique queries.")

    # Try direct (query, answer) tuple
    val = pos_tuple_map.get((query, answer))
    if val is not None:
        return val

    # Normalize
    qn = query.strip().lower()
    an = answer.strip().lower()

    qmap = pos_query_map.get(qn)
    if not qmap:
        return [0.0, 0.0]

    # Exact normalized answer match
    val = qmap.get(an)
    if val is not None:
        return val

    # Fuzzy match (substring)
    for cand_ans, v in qmap.items():
        if cand_ans in an or an in cand_ans:
            return v

    return [0.0, 0.0]


def load_qa_dataset_train(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    print(f"\n🔄 Loading dataset '{name}'")
    print(f"  ▶️ Positive pickle file: {pos_dir}")
    print(f"  ▶️ Negative pickle file: {neg_dir}")
    print(f"  ▶️ Data file (TSV/JSONL): {file_path}\n")

    data = []
    # =========================================
    # ✅ Cached Teacher Logit Loader (fast path)
    # =========================================
    cache_file = pos_dir
    if not os.path.exists(cache_file):
        print(f"🧠 Teacher logits cache not found: {cache_file}")
        print("🚀 Generating teacher logits once and caching to disk...")
        from tqdm import tqdm
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        # Load teacher model (adjust name if different)
        teacher_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_name).cuda().eval()

        # Read all queries/answers
        teacher_logits = {}
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in tqdm(reader, desc="Generating teacher logits"):
                text_a = row["sentence1"]
                text_b = row["sentence2"]

                inputs = teacher_tokenizer(
                    text_a, text_b, return_tensors="pt", truncation=True,
                    max_length=256, padding="max_length"
                ).to("cuda")

                with torch.no_grad():
                    outputs = teacher_model(**inputs)
                    logits = outputs.logits.squeeze().detach().cpu().numpy().tolist()

                teacher_logits[(text_a, text_b)] = logits

        # Save once
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        import pickle
        with open(cache_file, "wb") as f:
            pickle.dump(teacher_logits, f)
        print(f"✅ Teacher logits cached to {cache_file}")
        pos_logits = teacher_logits
    else:
        pos_logits = load_pickle(cache_file)
        print(f"✅ Loaded cached teacher logits from {cache_file} ({len(pos_logits)} entries)")

    hard_neg_house = load_pickle(neg_dir)

    # Counters
    total_records = 0
    with_hard_negs = 0
    with_pos_logits = 0
    missing_hard_neg_count = 0
    not_found_count = 0

    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for id, row in enumerate(reader):
            total_records += 1

            text_a = row['sentence1']
            text_b = row['sentence2']

            # Check for hard negatives
            neg_list = hard_neg_house.get(text_a, [])
            if not neg_list:
                missing_hard_neg_count += 1
                continue
            with_hard_negs += 1

            if len(neg_list) < neg_K:
                num = math.ceil(neg_K / len(neg_list))
                negs_logits = random.sample(neg_list * num, neg_K)
            else:
                negs_logits = random.sample(neg_list, neg_K)

            hardnegs, hardneg_logits = zip(*negs_logits)
            hardnegs = [sample[:320] for sample in hardnegs]
            hardneg_logits = list(hardneg_logits)

            # Lookup positive logit vector
            # ✅ Use flexible lookup just like QA loader
            pos_logit_for_sample = fast_lookup_pos_logit(pos_logits, text_a, text_b)

            # Ensure positive logits are numpy arrays (for consistent shape)
            pos_logit_for_sample = np.array(pos_logit_for_sample, dtype=np.float32)

            # Match embedding dimension with hard negatives
            if len(pos_logit_for_sample.shape) == 1 and len(pos_logit_for_sample) != len(hardneg_logits[0]):
                # Expand scalar or short vector to same dim as hardneg logits
                pos_logit_for_sample = np.full_like(hardneg_logits[0], float(np.mean(pos_logit_for_sample)))

            if np.all(pos_logit_for_sample == 0.0):
                not_found_count += 1
            else:
                with_pos_logits += 1

            data.append((text_a[:50], text_b[:320], pos_logit_for_sample.tolist(),
                         hardnegs, hardneg_logits, 1))

    # Stats reporting
    print("\n📊 Dataset Load Summary")
    print(f"   • Total records in input file: {total_records}")
    print(f"   • Records with hard negatives: {with_hard_negs}")
    print(f"   • Records with NO hard negatives: {missing_hard_neg_count}")
    print(f"   • Records with positive logits: {with_pos_logits}")
    print(f"   • Records missing positive logits: {not_found_count}")
    print(f"   • Final usable samples: {len(data)}\n")

    sample_num = len(data)
    res_data.extend(data)

    return res_data, sample_num



def load_qa_dataset_val(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = []
    skipped_count = 0

    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        for id, row in enumerate(reader):
            # Use .get to avoid KeyError if header mismatch
            text_a = row.get('sentence1')
            text_b = row.get('sentence2')

            # Replace None with empty string
            if text_a is None:
                text_a = ""
                skipped_count += 1
            if text_b is None:
                text_b = ""
                skipped_count += 1

            # Normalize the "No Answer Present." placeholder
            if text_b.strip().lower() == "no answer present.":
                text_b = ""

            # Safe slicing even if text_a/text_b == ""
            data.append((text_a[:50], text_b[:320], [], [], [], 1))

    if skipped_count:
        print(f"⚠️ {skipped_count} val samples had missing fields, replaced with empty string.")

    sample_num = len(data)
    res_data.extend(data)
    return res_data, sample_num


def load_mmarco_dataset_train(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"\n🔄 Loading dataset '{name}'")
    print(f"  ▶️ Positive pickle file: {pos_dir}")
    print(f"  ▶️ Negative pickle file: {neg_dir}")
    print(f"  ▶️ Data file (TSV/JSONL): {file_path}\n")

    data = []

    # ==========================================================
    # 🧠 Step 1: Ensure teacher logits exist or generate them
    # ==========================================================
    if not os.path.exists(pos_dir):
        print(f"🧠 Teacher logits cache not found: {pos_dir}")
        print("🚀 Generating teacher logits using cross-encoder/ms-marco-MiniLM-L-6-v2 ...")

        teacher_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_name).cuda().eval()

        teacher_logits = {}
        with open(file_path, encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in tqdm(reader, desc="Generating teacher logits for MS MARCO"):
                text_a = row['sentence1']
                text_b = row['sentence2']

                inputs = teacher_tokenizer(
                    text_a, text_b,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256,
                    padding='max_length'
                ).to('cuda')

                with torch.no_grad():
                    logits = teacher_model(**inputs).logits.squeeze().cpu().numpy().tolist()

                teacher_logits[(text_a, text_b)] = logits

        # Save to pickle
        os.makedirs(os.path.dirname(pos_dir), exist_ok=True)
        with open(pos_dir, 'wb') as f:
            pickle.dump(teacher_logits, f)

        pos_logits = teacher_logits
        print(f"✅ Generated and cached teacher logits for {len(pos_logits)} samples at {pos_dir}")
    else:
        pos_logits = load_pickle(pos_dir)
        print(f"✅ Loaded cached teacher logits from {pos_dir} ({len(pos_logits)} entries)")

    # ==========================================================
    # 🧩 Step 2: Load hard negatives (from teacher scores)
    # ==========================================================
    hard_neg_house = load_pickle(neg_dir)

    # Counters
    total_records = 0
    with_hard_negs = 0
    with_pos_logits = 0
    missing_hard_neg_count = 0
    not_found_count = 0

    # ==========================================================
    # 📥 Step 3: Read the MS MARCO training pairs
    # ==========================================================
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for _, row in enumerate(reader):
            total_records += 1

            text_a = row['sentence1']
            text_b = row['sentence2']

            # Hard negatives
            neg_list = hard_neg_house.get(text_a, [])
            if not neg_list:
                missing_hard_neg_count += 1
                continue
            with_hard_negs += 1

            if len(neg_list) < neg_K:
                num = math.ceil(neg_K / len(neg_list))
                negs_logits = random.sample(neg_list * num, neg_K)
            else:
                negs_logits = random.sample(neg_list, neg_K)

            hardnegs, hardneg_logits = zip(*negs_logits)
            hardnegs = [sample[:320] for sample in hardnegs]
            hardneg_logits = list(hardneg_logits)

            # Positive teacher logits
            pos_logit_for_sample = fast_lookup_pos_logit(pos_logits, text_a, text_b)

            pos_logit_for_sample = np.array(pos_logit_for_sample, dtype=np.float32)
            if np.all(pos_logit_for_sample == 0.0):
                not_found_count += 1
            else:
                with_pos_logits += 1

            data.append(
                (text_a.strip(), text_b.strip(), pos_logit_for_sample.tolist(), hardnegs, hardneg_logits, 1)
            )

    # ==========================================================
    # 📊 Step 4: Print dataset summary
    # ==========================================================
    print("\n📊 Dataset Load Summary")
    print(f"   • Total records in input file: {total_records}")
    print(f"   • Records with hard negatives: {with_hard_negs}")
    print(f"   • Records with NO hard negatives: {missing_hard_neg_count}")
    print(f"   • Records with positive logits: {with_pos_logits}")
    print(f"   • Records missing positive logits: {not_found_count}")
    print(f"   • Final usable samples: {len(data)}\n")

    sample_num = len(data)
    res_data.extend(data)
    return res_data, sample_num


def load_mmarco_dataset_val(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    data = []
    skipped_count = 0

    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        for id, row in enumerate(reader):
            # Use .get to avoid KeyError if header mismatch
            text_a = row.get('sentence1')
            text_b = row.get('sentence2')

            # Replace None with empty string
            if text_a is None:
                text_a = ""
                skipped_count += 1
            if text_b is None:
                text_b = ""
                skipped_count += 1

            # Normalize the "No Answer Present." placeholder
            if text_b.strip().lower() == "no answer present.":
                text_b = ""

            # Safe slicing even if text_a/text_b == ""
            data.append((text_a[:50], text_b[:320], [], [], [], 1))

    if skipped_count:
        print(f"⚠️ {skipped_count} val samples had missing fields, replaced with empty string.")

    sample_num = len(data)
    res_data.extend(data)
    return res_data, sample_num

def collate_fn(data):
    res_s_a = []
    res_s_b = []
    res_pos_logits = []
    res_neg_K = []
    res_neg_logits = []
    res_task_id = []

    for d in data[0]:
        res_s_a.append(d[0])
        res_s_b.append(d[1])
        res_pos_logits.append(d[2])
        res_neg_K.append(d[3])
        res_neg_logits.extend(d[4])
        res_task_id.append(int(d[5]))

    res_neg_K = [list(group) for group in zip(*res_neg_K)]
    res_neg_K = [e for l in res_neg_K for e in l]


    # Debug prints for pos logits
    # print("🟢 Total positive logits collected:", len(res_pos_logits))
    # if len(res_pos_logits) > 0:
        # print("🔍 Type of first positive logit sample:", type(res_pos_logits[0]))
        # print("📏 Length of first positive logit sample:", len(res_pos_logits[0]))
        # If list or numpy array, print length and first few values
        # if isinstance(res_pos_logits[0], (list, np.ndarray)):
        #     print("📏 Length of first positive logit sample:", len(res_pos_logits[0]))
        #     print("👀 First positive logit sample (first 10 elements):", res_pos_logits[0][:10])
        # else:
        #     print("👀 First positive logit sample value:", res_pos_logits[0])

    # print("🔴 Total negative logits collected:", len(res_neg_logits))

    return res_s_a, res_s_b, torch.FloatTensor(res_pos_logits), res_neg_K, torch.FloatTensor(res_neg_logits), torch.LongTensor(res_task_id)

    # return res_s_a, res_s_b, torch.FloatTensor(np.array(res_pos_logits, dtype=np.float32)), res_neg_K, torch.FloatTensor(res_neg_logits), torch.LongTensor(res_task_id)



class TrainDataset(Dataset):

    def __init__(self, tokenizer, pos_dir, neg_dir, datadir, names=None, batch_size=32, neg_K=8, process_index=0,
                 num_processes=1, seed=2023):
        self.dataset_id_dict = DATASET_ID_DICT
        self.tokenizer = tokenizer
        self.data = []
        self.batch_size = batch_size
        self.sample_stas = dict()
        self.dataset_indices_range = dict()
        self.process_index = process_index
        self.num_processes = num_processes
        self.neg_K = neg_K
        self.deterministic_generator = np.random.default_rng(seed)
        names.sort(reverse=True)
        for name in names:
            if name in ['snli']:
                if name == 'snli':
                    start_id = len(self.data)
                    self.data, sample_num = load_text_dataset(name, os.path.join(pos_dir, 'pos_emb/snli_train_pos_emb.pkl'),
                                                              os.path.join(neg_dir, 'logits/snli_train_logits.pkl'),
                                                              os.path.join(datadir, 'snli_1.0/snli_1.0_train.tsv'), self.neg_K,
                                                              self.data, 'train')
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            elif name in ['sts']:
                if name == 'sts':
                    start_id = len(self.data)
                    self.data, sample_num = load_sts_dataset_train(name, os.path.join(pos_dir, 'pos_emb/sts_train_pos_emb.pkl'),
                                                                   os.path.join(neg_dir, 'logits/sts_train_logits.pkl'),
                                                                   os.path.join(datadir, 'sts/train.csv'),
                                                                   self.neg_K, self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            elif name in ['t2', 'du', 'mmarco', 'wq']:
                if name == 'mmarco':
                    start_id = len(self.data)
                    self.data, sample_num = load_mmarco_dataset_train(name, os.path.join(pos_dir, 'pos_emb/mmarco_train_pos_emb.pkl'),
                                                                      os.path.join(neg_dir, 'logits/mmarco_train_logits.pkl'),
                                                                      os.path.join(datadir, 'ms_marco/train/positives.tsv'), self.neg_K,
                                                                      self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num

                elif name == 'wq':
                    start_id = len(self.data)
                    self.data, sample_num = load_qa_dataset_train(name, os.path.join(pos_dir, 'pos_emb/webq_train_pos_emb.pkl'),
                                                                      os.path.join(neg_dir, 'logits/webq_train_logits.pkl'),
                                                                      os.path.join(datadir, 'web_questions/train/positives.tsv'), self.neg_K,
                                                                      self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            else:
                logger.debug('Unknown dataset: {}'.format(name))



        print(f"[DEBUG] TrainDataset initialized with {len(self.data)} samples.")
        self.create_epoch()

    def __len__(self):
        return self.steps_per_epoch * self.num_processes

    def create_epoch(self):
        epoch = []
        self.steps_per_epoch = 0
        for k, v in self.dataset_indices_range.items():
            dataset_range = np.arange(*v)
            num_batches, remainer = divmod(len(dataset_range), self.batch_size * self.num_processes)
            if remainer != 0:
                dataset_range = dataset_range[:num_batches * self.batch_size * self.num_processes]
            self.deterministic_generator.shuffle(dataset_range)
            batches = dataset_range.reshape(num_batches * self.num_processes, self.batch_size).tolist()
            epoch.extend(batches)
            self.steps_per_epoch += num_batches
        self.deterministic_generator.shuffle(epoch)
        self.epoch = epoch
        self.step = 0

    def __getitem__(self, index: int):
        if self.step > (self.steps_per_epoch - 1):
            self.step = 0
        batch_indices = self.epoch[self.step * self.num_processes + self.process_index]
        batch_data = [self.data[i] for i in batch_indices]
        self.step += 1

        return batch_data


# --- DEBUG CHECK: Inspect one batch to verify positives vs hard negatives ---
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = TrainDataset(
        tokenizer=tokenizer,
        pos_dir="/root/pycharm_semanticsearch/outputs",  # <-- use actual absolute path
        neg_dir="/root/pycharm_semanticsearch/outputs",  # <-- actual path
        datadir="/root/pycharm_semanticsearch/dataset",               # <-- actual data root
        names=["mmarco"],
        batch_size=2,
        neg_K=2,
    )

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    for batch in loader:
        queries, positives, pos_logits, hardnegs, hardneg_logits, task_ids = batch
        print("🔹 Query:", queries[0])
        print("✅ Positive:", positives[0])
        print("❌ Hard Negatives:", hardnegs[:2])
        print("📈 Pos Logit:", pos_logits[0].tolist() if len(pos_logits) > 0 else "N/A")
        import numpy as np
        print("🔍 Hardneg logits example:", np.array(hardneg_logits[0]) if len(hardneg_logits) > 0 else "N/A")

        break

class ValDataset(Dataset):

    def __init__(self, tokenizer, pos_dir, neg_dir, datadir, names=None, batch_size=32, neg_K=8, process_index=0,
                 num_processes=1, seed=2023):
        self.dataset_id_dict = DATASET_ID_DICT
        self.tokenizer = tokenizer
        self.data = []
        self.batch_size = batch_size
        self.neg_K = neg_K
        self.sample_stas = dict()
        self.dataset_indices_range = dict()
        self.process_index = process_index
        self.num_processes = num_processes
        self.deterministic_generator = np.random.default_rng(seed)
        names.sort(reverse=True)
        for name in names:
            if name in ['snli']:
                if name == 'snli':
                    start_id = len(self.data)
                    self.data, sample_num = load_text_dataset(name, os.path.join(pos_dir, 'pos_emb/snli_val_pos_emb.pkl'),
                                                              os.path.join(neg_dir, 'logits/snli_val_logits.pkl'),
                                                              os.path.join(datadir, 'snli_1.0/snli_1.0_dev.tsv'), self.neg_K,
                                                              self.data, 'validation')
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            elif name in ['sts']:
                if name == 'sts':
                    start_id = len(self.data)
                    self.data, sample_num = load_sts_dataset_val(name, os.path.join(pos_dir, 'pos_emb/sts_val_pos_emb.pkl'),
                                                                 os.path.join(neg_dir, 'logits/sts_val_logits.pkl'),
                                                                 os.path.join(datadir, 'sts/validation.csv'), self.neg_K,
                                                                 self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            elif name in ['t2', 'du', 'mmarco', 'wq']:

                if name == 'mmarco':
                    start_id = len(self.data)
                    self.data, sample_num = load_mmarco_dataset_val(name, os.path.join(pos_dir, 'pos_emb/mmarco_val_pos_emb.pkl'),
                                                                    os.path.join(neg_dir, 'logits/mmarco_val_logits.pkl'),
                                                                    os.path.join(datadir, 'ms_marco/val/positives.tsv'), self.neg_K,
                                                                    self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
                elif name == 'wq':
                    start_id = len(self.data)
                    self.data, sample_num = load_qa_dataset_train(name, os.path.join(pos_dir, 'pos_emb/webq_val_pos_emb.pkl'),
                                                                      os.path.join(neg_dir, 'logits/webq_val_logits.pkl'),
                                                                      os.path.join(datadir, 'web_questions/val/positives.tsv'), self.neg_K,
                                                                      self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
            else:
                logger.debug('Unknown dataset: {}'.format(name))
        self.create_epoch()

    def __len__(self):
        return self.steps_per_epoch * self.num_processes

    def create_epoch(self):
        epoch = []
        self.steps_per_epoch = 0
        for k, v in self.dataset_indices_range.items():
            dataset_range = np.arange(*v)
            num_batches, remainer = divmod(len(dataset_range), self.batch_size * self.num_processes)
            if remainer != 0:
                dataset_range = dataset_range[:num_batches * self.batch_size * self.num_processes]
            self.deterministic_generator.shuffle(dataset_range)
            batches = dataset_range.reshape(num_batches * self.num_processes, self.batch_size).tolist()
            epoch.extend(batches)
            self.steps_per_epoch += num_batches
        self.deterministic_generator.shuffle(epoch)
        self.epoch = epoch
        self.step = 0

    def __getitem__(self, index: int):

        if self.step > self.steps_per_epoch - 1:
            self.step = 0
        batch_indices = self.epoch[self.step * self.num_processes + self.process_index]
        batch_data = [self.data[i] for i in batch_indices]
        self.step += 1
        return batch_data