import os
import torch
import optuna
from loss_calc import ci_loss, ri_loss, fi_loss
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer
from models import LoRAStudentModel
from evaluate import evaluate_student_model
import math
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import logging
import warnings
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def load_preprocessed_data(file_path):
    df = pd.read_excel(file_path)

    sentences1 = df["Sentence1"].tolist()
    sentences2 = df["Sentence2"].tolist()
    labels = df["Label"].tolist()

    sentences = list(zip(sentences1, sentences2))

    unique_labels_before = set(labels)
    # print(f"Unique labels before mapping: {unique_labels_before}")

    label_mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}

    invalid_labels = [label for label in labels if label not in label_mapping]
    if invalid_labels:
        print(f"Warning: Found invalid labels: {set(invalid_labels)}")

    labels = [label_mapping.get(label, -1) for label in labels]

    unique_labels_after = set(labels)
    # print(f"Unique labels after mapping: {unique_labels_after}")

    return sentences, labels


def prepare_dataloader(sentences, labels, batch_size=500, max_seq_length=256):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    s1_list, s2_list = zip(*sentences)

    inputs = tokenizer(list(s1_list), list(s2_list), padding=True, truncation=True,
                       max_length=max_seq_length, return_tensors='pt')
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels_tensor)
    dataloader = DataLoader(dataset,shuffle=False, batch_size=batch_size)
    return dataloader


def load_teacher_logits(teacher_logits_path):
    """
    Loads the precomputed teacher logits from the given file.
    Args:
        teacher_logits_path: Path to the file containing precomputed teacher logits.
    """
    loaded = torch.load(teacher_logits_path)
    teacher_logits = loaded['logits']
    final_epoch_labels = loaded['labels']
    print(f"Teacher logits loaded from {teacher_logits_path}")
    return teacher_logits


def train_student_model(student_model, teacher_logits, dataloader, device, temperature=0.5, alpha=0.5, beta=0.01, gamma=0.1, num_epochs=30):
    print("\nTraining the Student Model with Teacher Logits and KD Losses...")
    student_model.to(device)
    student_model.train()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

    # Initialize a list to store logits after the final epoch
    all_logits = []

    for epoch in range(num_epochs):  # Use num_epochs as specified in the function call
        total_loss = 0
        total_ci_loss = 0
        total_rank_alignment_loss = 0
        total_hard_vs_easy_loss = 0
        total_fi_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            optimizer.zero_grad()

            # Select the teacher logits corresponding to the current batch
            batch_size_actual = input_ids.size(0)
            start_idx = batch_idx * batch_size_actual
            end_idx = start_idx + batch_size_actual
            teacher_logits_batch = teacher_logits[start_idx:end_idx]

            # Ensure teacher_logits_batch is a PyTorch tensor
            if isinstance(teacher_logits_batch, np.ndarray):
                teacher_logits_batch = torch.tensor(teacher_logits_batch, dtype=torch.float32).to(device)

            # Ensure teacher_logits_batch is shaped [batch_size, num_classes]
            if teacher_logits_batch.ndimension() == 1:  # If it's 1D, repeat to match batch size
                teacher_logits_batch = teacher_logits_batch.unsqueeze(0).repeat(input_ids.size(0), 1)
            elif teacher_logits_batch.ndimension() == 2 and teacher_logits_batch.size(0) != input_ids.size(0):
                # If teacher_logits_batch is 2D but batch size doesn't match, repeat to match
                teacher_logits_batch = teacher_logits_batch.repeat(input_ids.size(0), 1)

            output = student_model(input_ids, attention_mask=attention_mask)
            # print(f"Output type: {type(output)}")
            # print(f"Output length: {len(output)}")
            # print(f"Output content types: {[type(o) for o in output]}")

            student_logits, pooled_output, hidden_states = student_model(input_ids, attention_mask=attention_mask)

            # Debugging: Print the shapes of logits
            # print(f"Teacher logits shape: {teacher_logits_batch.shape}")
            # print(f"Student logits shape: {student_logits.shape}")

            batch_size, num_classes_teacher = teacher_logits_batch.shape
            batch_size_student, num_classes_student = student_logits.shape

            if batch_size != batch_size_student:
                print(f"Warning: Batch sizes do not match between teacher logits and student! Skipping this batch.")
                continue

            if num_classes_teacher != num_classes_student:
                print(f"Warning: Number of classes between teacher logits and student do not match! Skipping this batch.")
                continue

            # Create positive and hard negative masks and move them to the same device
            positive_mask = (labels == 0).float().to(device)  # entailment
            hard_negative_mask = (labels == 2).float().to(device)  # contradiction

            # CI Loss
            ci_loss_value = ci_loss(teacher_logits_batch, student_logits, positive_mask, hard_negative_mask, temperature)

            # RI_Loss
            rank_alignment_loss, hard_vs_easy_loss = ri_loss(teacher_logits_batch, student_logits, positive_mask,
                                                             hard_negative_mask, return_hard_vs_easy=True)

            # FI Loss
            fi_loss_value = fi_loss(teacher_logits_batch, student_logits, positive_mask, hard_negative_mask)

            # Combine losses with weights
            loss = ci_loss_value + alpha * rank_alignment_loss + beta * hard_vs_easy_loss + gamma * fi_loss_value

            if torch.isnan(loss).any():
                print(f"Warning: NaN detected in loss at epoch {epoch+1}, batch {batch_idx+1}. Skipping this batch.")
                continue

            loss.backward()

            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            total_ci_loss += ci_loss_value.item()
            total_rank_alignment_loss += rank_alignment_loss.item()
            total_hard_vs_easy_loss += hard_vs_easy_loss.item()
            total_fi_loss += fi_loss_value.item()

            # After the final epoch, save the logits
            if epoch == num_epochs - 1:
                all_logits.append(student_logits.detach().cpu().numpy())

        if math.isnan(total_loss):
            print(f"Warning: NaN detected in total loss at epoch {epoch+1}. Stopping training early.")
            break

        # print(f"\nEpoch {epoch + 1} Summary:")
        print(f"Total Loss {epoch + 1}: {total_loss:.4f}")
        # print(f"CI Loss: {total_ci_loss:.4f}")
        # print(f"Rank Alignment Loss: {total_rank_alignment_loss:.4f}")
        # print(f"Hard vs Easy Loss: {total_hard_vs_easy_loss:.4f}")
        # print(f"FI Loss: {total_fi_loss:.4f}")

    print("Training completed.")


def evaluate_teacher_logits(teacher_logits, labels):
    # teacher_logits: tensor of shape (N, num_classes)
    # labels: list or tensor of true labels

    data = torch.load('./teacher_model/teacher_logits.pt')
    logits = data['logits']
    labels = data['labels']
    preds = torch.argmax(logits, dim=1)
    accuracy = (preds == labels).float().mean().item()
    print(f"Recovered Accuracy from saved logits: {accuracy:.4f}")


def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()
    inputs_list = []
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(logits, tuple):
                logits = logits[0]  # extract the actual logits tensor
            preds = torch.argmax(logits, dim=1)

            inputs_list.extend(input_ids.cpu().tolist())
            preds_list.extend(preds.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

            if len(preds_list) >= num_samples:
                break

    print("\nSample predictions:")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    for i in range(num_samples):
        input_ids = inputs_list[i]
        pred_label = preds_list[i]
        true_label = labels_list[i]

        # Decode token IDs back to text (only input_ids for sentence 1 for simplicity)
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"Input Text: {text}")
        print(f"Predicted Label: {pred_label}, True Label: {true_label}\n")


def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 2e-5, 5e-5, log=True)
    temperature = trial.suggest_float("temperature", 0.5, 2.0)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    beta = trial.suggest_float("beta", 0.0, 0.1)
    gamma = trial.suggest_float("gamma", 0.0, 0.1)
    num_epochs = 5  # Use smaller epochs for tuning

    # Load teacher logits
    teacher_logits_path = './teacher_model/teacher_logits.pt'
    loaded = torch.load(teacher_logits_path)
    teacher_logits = loaded['logits']

    # Initialize student model and move to device
    num_classes = 3
    student_model = LoRAStudentModel(model_name='bert-base-uncased', num_classes=num_classes)
    student_model.to(device)

    # Prepare data
    data_file = '../preprocess/preprocessed_data.xlsx'
    sentences, labels = load_preprocessed_data(data_file)
    dataloader = prepare_dataloader(sentences, labels)

    # Define optimizer with trial's learning rate
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)

    # Training loop (modified to accept Optuna's parameters)
    student_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            input_ids, attention_mask, labels_batch = [item.to(device) for item in batch]
            optimizer.zero_grad()

            batch_size_actual = input_ids.size(0)
            start_idx = batch_idx * batch_size_actual
            end_idx = start_idx + batch_size_actual
            teacher_logits_batch = teacher_logits[start_idx:end_idx].to(device)

            student_logits, _, _ = student_model(input_ids, attention_mask=attention_mask)

            # Loss calculations (simplified)
            positive_mask = (labels_batch == 0).float().to(device)
            hard_negative_mask = (labels_batch == 2).float().to(device)

            ci = ci_loss(teacher_logits_batch, student_logits, positive_mask, hard_negative_mask, temperature)
            ri, hve = ri_loss(teacher_logits_batch, student_logits, positive_mask, hard_negative_mask, return_hard_vs_easy=True)
            fi = fi_loss(teacher_logits_batch, student_logits, positive_mask, hard_negative_mask)

            loss = ci + alpha * ri + beta * hve + gamma * fi

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # You can optionally evaluate accuracy here and return that instead
    return avg_loss


def main(use_optuna=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load precomputed teacher logits
    teacher_logits_path = './teacher_model/teacher_logits.pt'
    if os.path.exists(teacher_logits_path):
        print(f"Loading teacher logits from {teacher_logits_path}...")
        teacher_logits = load_teacher_logits(teacher_logits_path)
    else:
        print(f"Teacher logits file not found at {teacher_logits_path}.")
        return

    num_classes = 3
    student_model = LoRAStudentModel(model_name='bert-base-uncased', num_classes=num_classes)

    student_model.to(device)

    data_file = '../preprocess/preprocessed_data.xlsx'
    sentences, labels = load_preprocessed_data(data_file)
    dataloader = prepare_dataloader(sentences, labels)

    # Run Optuna only if desired
    if use_optuna:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20)

        print("Best hyperparameters: ", study.best_params)
        print("Best loss: ", study.best_value)

        # Use best parameters in final training
        best_params = study.best_params
        train_student_model(
            student_model,
            teacher_logits,
            dataloader,
            device,
            temperature=best_params['temperature'],
            alpha=best_params['alpha'],
            beta=best_params['beta'],
            gamma=best_params['gamma'],
            num_epochs=30  # or customize further
        )
    else:
        # Default training
        train_student_model(student_model, teacher_logits, dataloader, device)

    evaluate_student_model(student_model, dataloader, device)

    print("\nStudent model training completed.")

    model_save_path = "./student_model/final_student_model.pt"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(student_model.state_dict(), model_save_path)
    print(f"Student model saved to {model_save_path}")

    # evaluate_teacher_logits(teacher_logits, labels)
    # visualize_predictions(student_model, dataloader, device, num_samples=5)


if __name__ == "__main__":
    main(use_optuna=True)  # Set to False to skip Optuna
