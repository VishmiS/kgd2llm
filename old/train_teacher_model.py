import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from old.evaluate import evaluate_teacher_model
import random
import numpy as np
import os
from sklearn.model_selection import train_test_split
from collections import Counter
from transformers import get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from transformers import GPT2Model

def set_random_seed(seed=42):
    """
    Sets a random seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_preprocessed_data(file_path):
    """
    Loads and processes the dataset from an Excel file for multi-class classification.
    Args:
        file_path: Path to the Excel file containing the dataset.
    Returns:
        sentences: Combined sentences from "Sentence1" and "Sentence2".
        labels: Numerical labels for classification (mapped to 0, 1, or 2).
    """
    df = pd.read_excel(file_path)

    # Extract Sentence1, Sentence2, and Labels
    sentences1 = df["Sentence1"].tolist()
    sentences2 = df["Sentence2"].tolist()
    labels = df["Label"].tolist()

    # Combine Sentence1 and Sentence2 for input
    sentences = [f"{s1}\n{s2}" for s1, s2 in zip(sentences1, sentences2)]

    # Map labels to numerical values
    label_mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
    labels = [label_mapping.get(label, -1) for label in labels]

    return sentences, labels

def prepare_dataloader(sentences, labels, batch_size=32, max_seq_length=256):
    """
    Prepares the DataLoader for the dataset with tokenized sentences and corresponding labels.
    This function keeps data on CPU; device transfer happens in training loop.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=max_seq_length, return_tensors='pt')
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def create_prompts(sentences, prompt_type="sym"):
    """
    Generates prompts for the query-passage pairs.
    Args:
        sentences: List of combined query-passage pairs.
        prompt_type: Type of prompt, either "sym" for symmetric or "asym" for asymmetric.
    Returns:
        prompts: List of prompts concatenated with query-passage pairs.
    """
    if prompt_type == "sym":
        prompt = "Given the following two sentences, classify their relationship as entailment, neutral, or contradiction."
    else:  # "asym"
        prompt = "Does the query match the passage?"

    prompts = [f"{prompt} {sentence}" for sentence in sentences]
    return prompts

class GPT2CustomClassifier(nn.Module):
    def __init__(self, model_name='gpt2', num_labels=3):
        super(GPT2CustomClassifier, self).__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.gpt2.config.n_embd, num_labels)

        self.pad_token_id = self.gpt2.config.eos_token_id

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # Use the last non-padding token
        lengths = attention_mask.sum(dim=1) - 1  # subtract 1 for index
        pooled = last_hidden_state[torch.arange(last_hidden_state.size(0)), lengths]

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fn(logits, labels)

        return {'logits': logits, 'loss': loss} if loss is not None else {'logits': logits}

def train_teacher_model(teacher_model, dataloader, device, epochs=7, test_loader=None):
    """
    Trains the teacher model for multi-class classification.
    Args:
        teacher_model: The teacher model to train.
        dataloader: DataLoader containing the dataset.
        device: Device to run the model on (CPU or GPU).
        epochs: Number of epochs to train the model.
        num_classes: Number of classes in the classification task.
    """
    print("\nTraining the Teacher Model for Multi-Class Classification...")

    best_accuracy = 0.0
    best_model_state = None

    # Move the model to the device (GPU/CPU)
    teacher_model.to(device)
    teacher_model.train()

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(teacher_model.parameters(), lr=5e-5)
    total_steps = epochs * len(dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10,
                                                num_training_steps=total_steps)
    # Flatten all labels from the dataloader dataset (CPU tensors)
    all_labels_cpu = [label.item() for _, _, label in dataloader.dataset]
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_labels_cpu), y=all_labels_cpu)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    teacher_logits_storage = None  # Will store logits from the final epoch

    # Ensure tokenizer and model padding token compatibility
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # Use EOS as padding token if none is defined
    teacher_model.config.pad_token_id = tokenizer.pad_token_id

    for epoch in range(epochs):
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, batch in enumerate(dataloader):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Pass labels to model to get loss directly
            outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            logits = outputs['logits']

            loss.backward()

            torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.detach().cpu().numpy())

        # Store logits from the last epoch as a tensor on CPU
        # Accumulate logits and labels across all batches in the final epoch
        if epoch == epochs - 1:
            if teacher_logits_storage is None:
                teacher_logits_storage = logits.detach().cpu()
                final_epoch_labels = labels.detach().cpu()
            else:
                teacher_logits_storage = torch.cat([teacher_logits_storage, logits.detach().cpu()], dim=0)
                final_epoch_labels = torch.cat([final_epoch_labels, labels.detach().cpu()], dim=0)

        # Compute epoch accuracy
        epoch_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        if test_loader:
            teacher_model.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids, attention_mask, labels = [x.to(device) for x in batch]
                    outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_labels.extend(labels.cpu().numpy())

            val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
            print(f"Validation Accuracy: {val_acc:.4f}")
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model_state = teacher_model.state_dict()

    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        best_model_state = teacher_model.state_dict()

    print("\nTeacher model training completed.")

    # Save the trained model
    model_dir = './teacher_model'
    os.makedirs(model_dir, exist_ok=True)

    model_save_path = os.path.join(model_dir, 'teacher_model.pth')
    teacher_model.load_state_dict(best_model_state)
    torch.save(teacher_model.state_dict(), model_save_path)
    print(f"Best model (Accuracy: {best_accuracy:.4f}) saved to {model_save_path}")

    # Return logits from the final epoch for knowledge distillation or further use
    return teacher_logits_storage, final_epoch_labels


def main():
    # Set random seed for reproducibility
    set_random_seed(42)

    # Define device upfront
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load preprocessed data
    data_file = '../preprocess/preprocessed_data.xlsx'
    sentences, labels = load_preprocessed_data(data_file)
    print("Label distribution:", Counter(labels))

    # Set this flag to True if you want to train on the full dataset
    train_on_full_data = False

    if train_on_full_data:
        train_loader = prepare_dataloader(sentences, labels)
        test_loader = None  # No test evaluation in this case
    else:
        # Train/test split
        train_sentences, test_sentences, train_labels, test_labels = train_test_split(
            sentences, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print("Train label distribution:", Counter(train_labels))
        print("Test label distribution:", Counter(test_labels))

        # Prepare dataloaders
        train_loader = prepare_dataloader(train_sentences, train_labels)
        test_loader = prepare_dataloader(test_sentences, test_labels)

    # Initialize GPT-2 model for classification
    num_classes = 3
    teacher_model = GPT2CustomClassifier(model_name="gpt2", num_labels=num_classes)

    # Train and save teacher logits
    teacher_logits_storage, final_epoch_labels = train_teacher_model(teacher_model, train_loader, device, epochs=7, num_classes=num_classes)
    model_dir = './teacher_model'
    os.makedirs(model_dir, exist_ok=True)
    logits_save_path = os.path.join(model_dir, 'teacher_logits.pt')
    torch.save({
        'logits': teacher_logits_storage,
        'labels': final_epoch_labels
    }, logits_save_path)
    print(f"Teacher logits from the final epoch saved to {logits_save_path}")

    # Evaluate the teacher model
    if test_loader:
        print("\nEvaluating on test set:")
        evaluate_teacher_model(teacher_model, test_loader, device)
    else:
        print("\nSkipped test evaluation (trained on full dataset).")

if __name__ == "__main__":
    main()
