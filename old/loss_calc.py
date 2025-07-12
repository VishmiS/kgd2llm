import os
import torch
import torch.nn as nn
from models import LoRAStudentModel
import torch.nn.functional as F


def ci_loss(teacher_logits, student_logits, positive_mask, hard_negative_mask, temperature=1.0):
    """
    Compute the cross-information (KL divergence) loss between teacher and student logits,
    applying masks to select relevant examples (positives and hard negatives).
    """
    device = teacher_logits.device

    # Check input shapes
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("Shape mismatch: teacher_logits and student_logits must have the same shape")

    batch_size, num_classes = teacher_logits.shape

    # Ensure masks are 1D and same batch size
    if positive_mask.dim() != 1 or hard_negative_mask.dim() != 1:
        raise ValueError("positive_mask and hard_negative_mask must be 1D (batch_size,)")
    if positive_mask.shape[0] != batch_size or hard_negative_mask.shape[0] != batch_size:
        raise ValueError("Mask sizes must match batch size")

    # Combine masks: examples to include in loss
    selected_mask = positive_mask.bool() | hard_negative_mask.bool()
    valid_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)

    if valid_indices.numel() == 0:
        print("Warning: No valid samples found for CI loss.")
        return torch.tensor(0.0, device=device)

    valid_indices = valid_indices.to(teacher_logits.device)

    # Select only the valid samples
    teacher_logits_sel = teacher_logits[valid_indices] / temperature
    student_logits_sel = student_logits[valid_indices] / temperature

    # Apply log-softmax and softmax
    teacher_probs = F.softmax(teacher_logits_sel, dim=-1)
    student_log_probs = F.log_softmax(student_logits_sel, dim=-1)

    teacher_probs = teacher_probs.to(student_log_probs.device)

    # KL divergence loss (student log-probs vs teacher probs)
    kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    if torch.isnan(kl_loss).any():
        print("Warning: NaN detected in CI loss!")
        return torch.tensor(0.0, device=device)

    return kl_loss



def ri_loss(teacher_logits, student_logits, positive_mask, hard_negative_mask, epsilon=1e-8, return_hard_vs_easy=False):
    """
    Compute Rank Imitation (RI) loss to align ranking and differentiate negatives.
    Args:
        teacher_logits: Logits from the teacher model (batch_size x num_classes).
        student_logits: Logits from the student model (batch_size x num_classes).
        positive_mask: Binary mask indicating positive samples (batch_size).
        hard_negative_mask: Binary mask indicating hard negatives (batch_size).
        epsilon: A small value to prevent division by zero in correlation computation.
        return_hard_vs_easy: Whether to return both the rank alignment loss and the hard vs easy loss.
    Returns:
        If `return_hard_vs_easy=True`, returns both rank alignment loss and hard vs easy loss.
        Otherwise, returns only the rank alignment loss.
    """
    # print("\nRI Loss calculating...")

    # Ensure all tensors are on the same device
    device = teacher_logits.device
    student_logits = student_logits.to(device)
    positive_mask = positive_mask.to(device)
    hard_negative_mask = hard_negative_mask.to(device)

    # Teacher logits should already have the correct shape [batch_size, num_classes]
    # print(f"Teacher Logits Shape: {teacher_logits.shape}")

    # No need to expand teacher logits; just ensure that its shape is [batch_size, num_classes]
    teacher_logits_expanded = teacher_logits  # [batch_size, num_classes]
    # print(f"Teacher Logits Expanded Shape: {teacher_logits_expanded.shape}")  # Debugging

    # Expand the masks to match the shape of student_logits
    positive_mask_expanded = positive_mask.unsqueeze(-1).expand(-1, student_logits.size(1))  # [batch_size, num_classes]
    hard_negative_mask_expanded = hard_negative_mask.unsqueeze(-1).expand(-1, student_logits.size(1))  # [batch_size, num_classes]

    # print(f"Positive Mask Expanded Shape: {positive_mask_expanded.shape}")  # Debugging
    # print(f"Hard Negative Mask Expanded Shape: {hard_negative_mask_expanded.shape}")  # Debugging

    # Apply the masks to logits
    student_logits_pos_hard = student_logits * (positive_mask_expanded + hard_negative_mask_expanded)  # [batch_size, num_classes]
    teacher_logits_pos_hard = teacher_logits_expanded * (positive_mask_expanded + hard_negative_mask_expanded)  # [batch_size, num_classes]

    # print(f"Student Logits with Positive/Hard Mask Shape: {student_logits_pos_hard.shape}")  # Debugging
    # print(f"Teacher Logits with Positive/Hard Mask Shape: {teacher_logits_pos_hard.shape}")  # Debugging

    # Centering the logits
    teacher_mean = teacher_logits_pos_hard.mean(dim=-1, keepdim=True)  # [batch_size, 1]
    student_mean = student_logits_pos_hard.mean(dim=-1, keepdim=True)  # [batch_size, 1]

    teacher_centered = teacher_logits_pos_hard - teacher_mean  # [batch_size, num_classes]
    student_centered = student_logits_pos_hard - student_mean  # [batch_size, num_classes]

    # Calculate correlation with epsilon to avoid division by zero
    teacher_norm = torch.sqrt(torch.sum(teacher_centered ** 2, dim=-1) + epsilon)  # [batch_size]
    student_norm = torch.sqrt(torch.sum(student_centered ** 2, dim=-1) + epsilon)  # [batch_size]
    correlation = torch.sum(teacher_centered * student_centered, dim=-1) / (teacher_norm * student_norm + epsilon)  # [batch_size]

    rank_alignment_loss = 1 - correlation.mean()

    # print(f"Rank Alignment Loss loss: {rank_alignment_loss.item()}")  # Debugging

    # Differentiating hard vs. easy negatives (if necessary)
    # Per sample, get scalar hardest negative and easiest negative scores.
    hard_scores = torch.topk(student_logits * hard_negative_mask_expanded, k=3, dim=1).values.mean(dim=1)

    easy_mask = 1 - positive_mask - hard_negative_mask
    easy_mask_exp = easy_mask.unsqueeze(-1).expand_as(student_logits)
    easy_scores = torch.topk(student_logits * easy_mask_exp, k=3, dim=1).values.mean(dim=1)

    # margin ranking loss:
    margin = torch.nn.Parameter(torch.tensor(1.0, device=device), requires_grad=True)
    hard_vs_easy_loss = torch.relu(margin + easy_scores - hard_scores).mean()

    # print(f"Hard vs Easy Loss loss: {hard_vs_easy_loss.item()}")  # Debugging

    if return_hard_vs_easy:
        return rank_alignment_loss, hard_vs_easy_loss
    else:
        return rank_alignment_loss


def fi_loss(teacher_embeddings, student_embeddings, positive_mask, hard_negative_mask):
    """
    Compute Feature Imitation (FI) loss using similarity matrices.
    Args:
        teacher_embeddings: Embeddings from teacher (batch_size x embed_dim).
        student_embeddings: Embeddings from student (batch_size x embed_dim).
        positive_mask: Binary mask indicating positive samples (batch_size).
        hard_negative_mask: Binary mask indicating hard negatives (batch_size).
    Returns:
        FI loss value.
    """
    # print("\nFI Loss calculating...")

    device = teacher_embeddings.device
    student_embeddings = student_embeddings.to(device)
    positive_mask = positive_mask.to(device)
    hard_negative_mask = hard_negative_mask.to(device)

    # Compute similarity matrices
    teacher_sim = F.cosine_similarity(
        teacher_embeddings.unsqueeze(1), teacher_embeddings.unsqueeze(0), dim=-1
    )  # [batch_size, batch_size]
    student_sim = F.cosine_similarity(
        student_embeddings.unsqueeze(1), student_embeddings.unsqueeze(0), dim=-1
    )  # [batch_size, batch_size]

    # Debug shapes
    # print(f"Teacher Sim Shape: {teacher_sim.shape}")
    # print(f"Student Sim Shape: {student_sim.shape}")

    # Create valid mask and expand
    valid_mask = (positive_mask + hard_negative_mask).unsqueeze(-1)  # [batch_size, 1]
    # print(f"Valid Mask Shape: {valid_mask.shape}")
    valid_mask_expanded = valid_mask.expand(-1, teacher_sim.size(1))  # [batch_size, batch_size]
    # print(f"Valid Mask Expanded Shape: {valid_mask_expanded.shape}")

    # Apply mask
    teacher_sim_valid = teacher_sim * valid_mask_expanded
    student_sim_valid = student_sim * valid_mask_expanded

    # print(f"Teacher Sim Valid Shape: {teacher_sim_valid.shape}")
    # print(f"Student Sim Valid Shape: {student_sim_valid.shape}")

    teacher_sim_valid = (teacher_sim_valid + 1) / 2
    student_sim_valid = (student_sim_valid + 1) / 2

    # Compute loss
    loss = F.mse_loss(teacher_sim_valid, student_sim_valid)
    # print(f"FI LOSS: {loss}")
    return loss


def ndcg_at_k(logits, k=10):
    """
    Compute NDCG at rank k for the logits.
    Args:
        logits: The logits or relevance scores [batch_size, num_classes].
        k: The rank at which to compute NDCG.
    Returns:
        NDCG score for each sample in the batch.
    """
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # Compute Discounted Cumulative Gain (DCG)
    gain = torch.pow(2, sorted_logits) - 1
    rank_position = torch.arange(2, gain.size(1) + 2, dtype=torch.float32, device=logits.device)
    dcg = torch.sum(gain / torch.log2(rank_position), dim=-1)

    # Compute Ideal DCG
    ideal_gain = torch.sort(logits, descending=True, dim=-1).values
    ideal_dcg = torch.sum(torch.pow(2, ideal_gain) - 1 / torch.log2(rank_position), dim=-1)

    # Compute NDCG
    ndcg = dcg / (ideal_dcg + 1e-8)

    return ndcg.mean()
