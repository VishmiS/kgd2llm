import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib
matplotlib.use('Agg') # non-interactive backend suitable for headless environments
import matplotlib.pyplot as plt
import seaborn as sns


# Evaluation Metrics
def calculate_accuracy(predictions, labels):
    # If predictions are already class indices, no need for argmax
    if predictions.ndimension() > 1:  # Handle logits
        predictions = torch.argmax(predictions, dim=-1)

    # Check and align shapes if necessary
    predictions = predictions.view(-1)  # Flatten predictions
    labels = labels.view(-1)  # Flatten labels

    accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
    return accuracy

def calculate_precision_recall_f1(predictions, labels):
    if predictions.ndimension() > 1:  # Handle logits
        predictions = torch.argmax(predictions, dim=-1)
    if predictions.size() != labels.size():
        raise ValueError(f"Shape mismatch: predictions={predictions.size()}, labels={labels.size()}")

    precision = precision_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro', zero_division=0)
    recall = recall_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro', zero_division=0)
    f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='macro', zero_division=0)
    return precision, recall, f1

def plot_confusion_matrix(predictions, labels, class_names=None):
    """
    Plots a confusion matrix for the predictions and labels.

    Args:
        predictions (torch.Tensor): Predicted class indices or logits.
        labels (torch.Tensor): True class indices.
        class_names (list): List of class names corresponding to indices.
    """
    # Ensure predictions are converted to class indices if they are logits
    if predictions.ndimension() > 1:
        predictions = torch.argmax(predictions, dim=-1)

    # Flatten predictions and labels to ensure consistency
    predictions = predictions.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()

    # Default class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(set(labels)))]

    # Compute confusion matrix, explicitly passing all class labels
    cm = confusion_matrix(labels, predictions, labels=range(len(class_names)))

    # Plotting the confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig("confusion_matrix.png")  # You can choose any path/filename
    plt.close()


def evaluate_teacher_model(teacher_model, dataloader, device):
    teacher_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids, attention_mask, labels = [item.to(device) for item in batch]

            # Forward pass through the teacher model
            outputs = teacher_model(input_ids, attention_mask=attention_mask)

            # Extract logits from the model outputs
            logits = outputs[0]  # Assuming logits are the first element of the output tuple

            # Get predictions (class with the highest probability)
            preds = torch.argmax(logits, dim=1)

            # Append the predictions and true labels
            all_preds.append(preds)
            all_labels.append(labels)

    # Combine all predictions and labels
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Debugging prints
    print(f"Labels shape: {all_labels.shape}")
    print(f"Predicted classes shape: {all_preds.shape}")
    print(f"Labels: {all_labels}")
    print(f"Predicted classes: {all_preds}")

    # Calculate accuracy, precision, recall, and F1 score
    accuracy = calculate_accuracy(all_preds, all_labels)
    precision, recall, f1 = calculate_precision_recall_f1(all_preds, all_labels)
    print(f"Teacher Model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Pass explicit class names to plot confusion matrix
    plot_confusion_matrix(all_preds, all_labels, class_names=["Class 0", "Class 1", "Class 2"])


def evaluate_student_model(student_model, dataloader, device):
    student_model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = [item.to(device) for item in batch]
            logits, *_ = student_model(input_ids, attention_mask=attention_mask)

            # Ensure the logits are 2D (batch_size, num_classes)
            if len(logits.shape) == 3:
                logits = logits[:, -1, :]  # If logits are 3D, take the last token's logits
            elif len(logits.shape) != 2:
                raise ValueError(f"Unexpected logits shape: {logits.shape}")

            # Get predicted classes (argmax over the class dimension)
            predicted_classes = torch.argmax(logits, dim=-1)
            all_preds.append(predicted_classes)
            all_labels.append(labels)

    # Combine all predictions and labels into tensors
    all_preds = torch.cat(all_preds).view(-1)
    all_labels = torch.cat(all_labels).view(-1)

    # Ensure shapes match
    assert all_preds.size() == all_labels.size(), f"Shape mismatch: {all_preds.size()} vs {all_labels.size()}"

    # Compute and print metrics
    accuracy = calculate_accuracy(all_preds, all_labels)
    precision, recall, f1 = calculate_precision_recall_f1(all_preds, all_labels)

    print(f"Student Model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Plot confusion matrix with explicit class names
    plot_confusion_matrix(all_preds, all_labels, class_names=["Class 0", "Class 1", "Class 2"])
