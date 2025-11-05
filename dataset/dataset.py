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

DATASET_ID_DICT = {'snli': 1, 'sts': 2, 'mmarco': 3, 'wq': 4, 'covid': 5}


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


def load_ir_dataset_train(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    print(f"Loading IR training dataset: {name}")
    print(f"  Positive logits cache: {pos_dir}")
    print(f"  Hard negatives cache: {neg_dir}")
    print(f"  Data file: {file_path}")

    data = []

    # Step 1: Load or generate teacher logits
    if not os.path.exists(pos_dir):
        print("Teacher logits cache not found, generating now...")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        teacher_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name)
        teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_name).cuda().eval()

        teacher_logits = {}
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                text_a = row["sentence1"].strip()
                text_b = row["sentence2"].strip()

                inputs = teacher_tokenizer(
                    text_a, text_b, return_tensors="pt", truncation=True,
                    max_length=256, padding="max_length"
                ).to("cuda")

                with torch.no_grad():
                    outputs = teacher_model(**inputs)
                    logits = outputs.logits.squeeze().detach().cpu().numpy().tolist()

                teacher_logits[(text_a, text_b)] = logits

        # Save cache
        os.makedirs(os.path.dirname(pos_dir), exist_ok=True)
        with open(pos_dir, "wb") as f:
            pickle.dump(teacher_logits, f)
        print(f"Generated and cached teacher logits for {len(teacher_logits)} samples")
        pos_logits = teacher_logits
    else:
        pos_logits = load_pickle(pos_dir)
        print(f"Loaded cached teacher logits: {len(pos_logits)} entries")

    # Step 2: Load hard negatives
    hard_neg_house = load_pickle(neg_dir)
    print(f"Loaded hard negatives: {len(hard_neg_house)} queries")

    # Counters for statistics
    total_records = 0
    with_hard_negs = 0
    with_pos_logits = 0
    missing_hard_neg_count = 0
    not_found_count = 0

    # Step 3: Process data samples
    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            total_records += 1

            text_a = row['sentence1'].strip()
            text_b = row['sentence2'].strip()

            # Check for hard negatives
            neg_list = hard_neg_house.get(text_a, [])
            if not neg_list:
                missing_hard_neg_count += 1
                continue
            with_hard_negs += 1

            # Sample hard negatives
            if len(neg_list) < neg_K:
                num = math.ceil(neg_K / len(neg_list))
                negs_logits = random.sample(neg_list * num, neg_K)
            else:
                negs_logits = random.sample(neg_list, neg_K)

            hardnegs, hardneg_logits = zip(*negs_logits)
            hardnegs = [sample.strip() for sample in hardnegs]
            hardneg_logits = list(hardneg_logits)

            # Get positive logits
            pos_logit_for_sample = fast_lookup_pos_logit(pos_logits, text_a, text_b)
            pos_logit_for_sample = np.array(pos_logit_for_sample, dtype=np.float32)

            if np.all(pos_logit_for_sample == 0.0):
                not_found_count += 1
            else:
                with_pos_logits += 1

            data.append((text_a, text_b, pos_logit_for_sample.tolist(), hardnegs, hardneg_logits, 1))

    # Step 4: Print detailed statistics
    print("Dataset Load Summary")
    print(f"  Total records in input file: {total_records}")
    print(f"  Records with hard negatives: {with_hard_negs}")
    print(f"  Records with NO hard negatives: {missing_hard_neg_count}")
    print(f"  Records with positive logits: {with_pos_logits}")
    print(f"  Records missing positive logits: {not_found_count}")
    print(f"  Final usable samples: {len(data)}")

    sample_num = len(data)
    res_data.extend(data)

    return res_data, sample_num


def load_ir_dataset_val(name, pos_dir, neg_dir, file_path, neg_K, res_data):
    print(f"Loading IR validation dataset: {name}")
    print(f"  Data file: {file_path}")

    data = []
    skipped_count = 0
    total_records = 0

    with open(file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)

        for row in reader:
            total_records += 1

            # Use .get to avoid KeyError if header mismatch
            text_a = row.get('sentence1')
            text_b = row.get('sentence2')

            # Handle missing fields
            if text_a is None:
                text_a = ""
                skipped_count += 1
            if text_b is None:
                text_b = ""
                skipped_count += 1

            # Clean and strip text
            text_a = text_a.strip()
            text_b = text_b.strip()

            # Normalize the "No Answer Present." placeholder
            if text_b.lower() == "no answer present.":
                text_b = ""

            # For validation, we don't need logits or hard negatives
            data.append((text_a, text_b, [], [], [], 1))

    # Print validation statistics
    print("Validation Dataset Load Summary")
    print(f"  Total records in input file: {total_records}")
    print(f"  Records with missing fields: {skipped_count}")
    print(f"  Final validation samples: {len(data)}")

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

    return res_s_a, res_s_b, torch.FloatTensor(res_pos_logits), res_neg_K, torch.FloatTensor(res_neg_logits), torch.LongTensor(res_task_id)



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
            elif name in ['t2', 'du', 'mmarco', 'wq', 'covid']:
                if name == 'mmarco':
                    start_id = len(self.data)
                    self.data, sample_num = load_ir_dataset_train(name, os.path.join(pos_dir, 'pos_emb/mmarco_train_pos_logits.pkl'),
                                                                      os.path.join(neg_dir, 'logits/mmarco_train_neg_logits.pkl'),
                                                                      os.path.join(datadir, 'ms_marco/train/positives.tsv'), self.neg_K,
                                                                      self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num

                elif name == 'wq':
                    start_id = len(self.data)
                    self.data, sample_num = load_ir_dataset_train(name, os.path.join(pos_dir, 'pos_emb/webq_train_pos_logits.pkl'),
                                                                      os.path.join(neg_dir, 'logits/webq_train_neg_logits.pkl'),
                                                                      os.path.join(datadir, 'web_questions/train/positives.tsv'), self.neg_K,
                                                                      self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
                elif name == 'covid':
                    start_id = len(self.data)
                    self.data, sample_num = load_ir_dataset_train(name, os.path.join(pos_dir, 'pos_emb/covid_train_pos_logits.pkl'),
                                                                      os.path.join(neg_dir, 'logits/covid_train_neg_logits.pkl'),
                                                                      os.path.join(datadir, 'covid/train/positives.tsv'), self.neg_K,
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
            elif name in ['t2', 'du', 'mmarco', 'wq', 'covid']:

                if name == 'mmarco':
                    start_id = len(self.data)
                    self.data, sample_num = load_ir_dataset_val(name, os.path.join(pos_dir, 'pos_emb/mmarco_val_pos_logits.pkl'),
                                                                    os.path.join(neg_dir, 'logits/mmarco_val_neg_logits.pkl'),
                                                                    os.path.join(datadir, 'ms_marco/val/positives.tsv'), self.neg_K,
                                                                    self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
                elif name == 'wq':
                    start_id = len(self.data)
                    self.data, sample_num = load_ir_dataset_val(name, os.path.join(pos_dir, 'pos_emb/webq_val_pos_logits.pkl'),
                                                                      os.path.join(neg_dir, 'logits/webq_val_neg_logits.pkl'),
                                                                      os.path.join(datadir, 'web_questions/val/positives.tsv'), self.neg_K,
                                                                      self.data)
                    end_id = len(self.data)
                    self.dataset_indices_range[self.dataset_id_dict[name]] = (start_id, end_id)
                    self.sample_stas[name] = sample_num
                elif name == 'covid':
                    start_id = len(self.data)
                    self.data, sample_num = load_ir_dataset_val(name, os.path.join(pos_dir, 'pos_emb/covid_val_pos_logits.pkl'),
                                                                      os.path.join(neg_dir, 'logits/covid_val_neg_logits.pkl'),
                                                                      os.path.join(datadir, 'covid/val/positives.tsv'), self.neg_K,
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


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    dataset = TrainDataset(
        tokenizer=tokenizer,
        pos_dir="/root/pycharm_semanticsearch/outputs",
        neg_dir="/root/pycharm_semanticsearch/outputs",
        datadir="/root/pycharm_semanticsearch/dataset",
        names=["covid"],
        batch_size=2,
        neg_K=8,
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
