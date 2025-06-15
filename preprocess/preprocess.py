import os
import json
import torch
import pandas as pd
import re
import time

from transformers import AutoTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Force NLTK data path
NLTK_DATA_PATH = "/root/nltk_data"
os.makedirs(NLTK_DATA_PATH, exist_ok=True)
nltk.data.path.append(NLTK_DATA_PATH)

# Download only what's needed
for resource in ['stopwords', 'wordnet', 'omw-1.4', 'punkt']:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=NLTK_DATA_PATH)

# HuggingFace tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Dataset path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SNLI_DIR = os.path.normpath(os.path.join(BASE_DIR, "../dataset/snli_1.0"))

# Custom tokenizer (no punkt required)
def custom_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Clean + tokenize + lemmatize
def preprocess_text(text, stop_words, lemmatizer):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = custom_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_snli_dataset():
    print("Loading SNLI dataset...")
    snli_files = [os.path.join(SNLI_DIR, f) for f in os.listdir(SNLI_DIR) if f.endswith('.jsonl')]
    snli_data = []
    for file in snli_files:
        print(f"Reading file: {file}")
        snli_data.extend(read_jsonl(file))
    print(f"Loaded {len(snli_data)} entries from SNLI dataset.")
    return snli_data

def preprocess_for_logit(data, max_samples=None, output_excel_file="preprocessed_data.xlsx", max_seq_length=256):
    print("Preprocessing and tokenizing data...")

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    processed_data = []
    invalid_count = 0
    sentences1, sentences2, labels = [], [], []

    for item in data:
        if max_samples and len(processed_data) >= max_samples:
            break

        label = item.get("gold_label")
        if label not in ["entailment", "neutral", "contradiction"]:
            invalid_count += 1
            continue

        sentence1 = item.get("sentence1")
        sentence2 = item.get("sentence2")

        if not sentence1 or not sentence2:
            invalid_count += 1
            continue

        sentence1 = preprocess_text(sentence1, stop_words, lemmatizer)
        sentence2 = preprocess_text(sentence2, stop_words, lemmatizer)

        sentences1.append(sentence1)
        sentences2.append(sentence2)
        labels.append(label)
        processed_data.append((sentence1, sentence2, label))

    print(f"Invalid labels skipped: {invalid_count}")
    print(f"Sample Sentence1: {sentences1[:1]}")
    print(f"Sample Sentence2: {sentences2[:1]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(sentences1, sentences2, padding=True, truncation=True, max_length=max_seq_length,
                       return_tensors='pt').to(device)

    label_mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
    label_ids = [label_mapping[label] for label in labels]
    labels_tensor = torch.tensor(label_ids).to(device)

    try:
        df = pd.DataFrame(processed_data, columns=["Sentence1", "Sentence2", "Label"])
        df.to_excel(output_excel_file, index=False)
        print(f"Preprocessed data saved to {output_excel_file}")
    except Exception as e:
        print(f"Error saving Excel file: {e}")

    return inputs, labels_tensor, sentences1, sentences2

# Entry point
if __name__ == "__main__":
    data = load_snli_dataset()
    start_time = time.time()
    inputs, labels_tensor, sentences1, sentences2 = preprocess_for_logit(data, max_samples=500)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Preprocessing completed in {elapsed_time:.2f} seconds.")
