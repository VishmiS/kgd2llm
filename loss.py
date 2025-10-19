import warnings
import logging
from utils.common_utils import *
import torch.nn.functional as F

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings('ignore')

def check_tensor(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        return
    if torch.isnan(tensor).any():
        print(f"[NaN DETECTED] in {name}")
    if torch.isinf(tensor).any():
        print(f"[INF DETECTED] in {name}")
    if tensor.abs().max() > 1e4:
        print(f"[LARGE VALUE DETECTED] in {name}, max: {tensor.abs().max().item()}")


def cal_loss_in_batch(args, student_logits, temperature, criterion=None):
    """
    Standard in-batch contrastive loss: cross-entropy over candidates.
    student_logits: [bs, num_candidates]
    """
    bs = student_logits.size(0)
    logits = student_logits / (temperature + 1e-12)
    labels = torch.zeros(bs, dtype=torch.long, device=logits.device)  # positive at idx 0

    # Use functional cross_entropy directly for stability
    loss = F.cross_entropy(logits, labels, reduction='mean')
    return loss


def cal_loss_hardneg(args, teacher_logits, student_logits, temperature_teacher, temperature, nll_criterion):
    """
    Hard negative loss between student and teacher.
    - teacher_logits: [bs, num_candidates, 2]
    - student_logits: [bs, num_candidates]
    - Uses teacher first channel for weighting and adjusts hard negatives.
    """
    loss_hardneg_weight = args.alpha
    bs = teacher_logits.size(0)
    num_candidates = teacher_logits.size(1)
    neg_K = num_candidates - 1

    if temperature_teacher <= 0:
        raise ValueError("temperature_teacher must be > 0!")

    # Softmax over candidates for teacher
    teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature_teacher, dim=1)

    # Take first channel (positive/primary score) for weighting
    teacher_probs_first = teacher_probs[..., 0]  # [bs, num_candidates]

    # Adjust hard negatives: invert teacher probability for all negatives except positive
    if neg_K > 0:
        teacher_probs_first[:, 1:] = 1.0 - teacher_probs_first[:, 1:]

    # Multiply student logits by teacher weights
    weighted_student_logits = student_logits * teacher_probs_first  # [bs, num_candidates]

    # Softmax + log for NLL
    student_log_probs = torch.nn.functional.log_softmax(weighted_student_logits / temperature, dim=-1)

    # Labels: all positives are at index 0
    labels = torch.zeros(bs, dtype=torch.long, device=student_logits.device)

    # Compute NLL loss
    loss_bs = nll_criterion(student_log_probs, labels)

    # Apply weight
    # Apply weight per-sample (do NOT detach)
    loss_bs = loss_bs * loss_hardneg_weight

    # Reduce by mean to maintain gradient flow
    loss_bs = loss_bs.mean()
    return loss_bs


def cal_loss_rd(args, teacher_logits, student_logits, teacher_temperature):
    loss_pearson_weight = args.beta

    # Ensure same dtype as model (bf16)
    dtype = student_logits.dtype

    teacher_probs = F.softmax(teacher_logits / teacher_temperature, dim=-1)[..., 0].to(dtype)
    student_probs = F.softmax(student_logits / teacher_temperature, dim=-1).to(dtype)

    teacher_probs = torch.clamp(teacher_probs, 1e-8, 1.0)
    student_probs = torch.clamp(student_probs, 1e-8, 1.0)


    loss = F.mse_loss(student_probs, teacher_probs) * loss_pearson_weight
    return loss


def cal_loss_rd2(args, teacher_logits_pos_hardneg, teacher_logits_pos_inbatch, teacher_temperature,
                 student_logits_pos_hardneg, student_logits_pos_inbatch, sigmoid, scale_param):
    loss_bpr_weight = args.gamma
    teacher_temperature = max(teacher_temperature, 1e-6)

    def softmax(X, temp):
        X = X / temp
        max_vals, _ = X.max(dim=-1, keepdims=True)
        X = X - max_vals
        X = X.exp()
        res = X / (X.sum(-1, keepdims=True) + 1e-20)
        return res

    teacher_logits_hardneg = softmax(teacher_logits_pos_hardneg, teacher_temperature)[:, 1:, 0]
    teacher_logits_inbatch = softmax(teacher_logits_pos_inbatch, teacher_temperature)[:, :, 0]

    bs = student_logits_pos_hardneg.size(0)
    neg_K = student_logits_pos_hardneg.size(1) - 1
    inbatch = student_logits_pos_inbatch.size(1) - 1

    student_logits_hardneg = student_logits_pos_hardneg[:, 1:]
    eye = torch.eye(bs, dtype=torch.bool, device=student_logits_pos_inbatch.device)
    student_logits_inbatch = student_logits_pos_inbatch[~eye].reshape(bs, -1)

    # print(f"[DEBUG] bs: {bs}, neg_K: {neg_K}, inbatch: {inbatch}")
    # print(f"[DEBUG] teacher_logits_hardneg.shape: {teacher_logits_hardneg.shape}")
    # print(f"[DEBUG] teacher_logits_inbatch.shape: {teacher_logits_inbatch.shape}")
    # print(f"[DEBUG] student_logits_hardneg.shape: {student_logits_hardneg.shape}")
    # print(f"[DEBUG] student_logits_inbatch.shape: {student_logits_inbatch.shape}")

    expanded_hardneg = student_logits_hardneg.view(bs, neg_K, 1).expand(-1, -1, inbatch).reshape(bs, -1)
    expanded_inbatch = student_logits_inbatch.unsqueeze(1).expand(-1, neg_K, -1).reshape(bs, -1)

    # print(f"[DEBUG] expanded_hardneg.shape: {expanded_hardneg.shape}")
    # print(f"[DEBUG] expanded_inbatch.shape: {expanded_inbatch.shape}")

    diff = expanded_hardneg - expanded_inbatch
    log_sigmoid_diff = F.logsigmoid(diff)

    # print(f"[DEBUG] diff: min={diff.min().item()}, max={diff.max().item()}, mean={diff.mean().item()}")
    # print(f"[DEBUG] sigmoid_diff: min={sigmoid_diff.min().item()}, max={sigmoid_diff.max().item()}, mean={sigmoid_diff.mean().item()}")
    # print(f"[DEBUG] log_sigmoid_diff: min={log_sigmoid_diff.min().item()}, max={log_sigmoid_diff.max().item()}, mean={log_sigmoid_diff.mean().item()}")

    loss_hardneg_inbatch = -log_sigmoid_diff

    # Clamp scale_param to avoid division by zero
    safe_scale_param = max(scale_param, 1e-8)
    if safe_scale_param != scale_param:
        print(f"[WARNING] scale_param clamped from {scale_param} to {safe_scale_param}")

    weight_hardneg_inbatch = teacher_logits_hardneg.repeat_interleave(inbatch, dim=1) - teacher_logits_inbatch.repeat((1, neg_K))
    weight_hardneg_inbatch = torch.clamp(weight_hardneg_inbatch, min=0) / safe_scale_param

    # Weighted per-sample loss
    loss_bs = (loss_hardneg_inbatch * weight_hardneg_inbatch).sum(-1)  # sum over candidates
    loss_bs = loss_bs * loss_bpr_weight

    # Take mean over batch for proper gradient flow
    loss_bs = loss_bs.mean()
    return loss_bs


def cal_feat_loss(args, teacher_feat_cos, student_feature_pos_hardneg):
    loss_feat_weight = args.eta
    neg_K = teacher_feat_cos.size(1)

    # Normalize features
    student_feature_pos_hardneg = student_feature_pos_hardneg / student_feature_pos_hardneg.norm(dim=-1, keepdim=True)

    # Compute cosine similarity matrix per batch
    student_feat_cos = torch.matmul(student_feature_pos_hardneg, student_feature_pos_hardneg.transpose(-2, -1))

    # Difference with teacher
    loss_bs = ((teacher_feat_cos - student_feat_cos) ** 2).sum((-1, -2))
    loss_bs = loss_bs * loss_feat_weight

    return loss_bs.sum() / (neg_K * neg_K)
