import warnings
import logging
from utils.common_utils import *

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


def cal_loss_in_batch(student_logits, temperature, criterion):
    bs = student_logits.size(0)
    logits = student_logits / temperature
    labels = torch.arange(bs, device=logits.device)

    # Assuming criterion returns scalar loss with default reduction='mean'
    loss = criterion(logits, labels)
    return loss


def cal_loss_hardneg(args, teacher_logits, student_logits, temperature_teacher, temperature, nll_criterion):
    loss_hardneg_weight = args.alpha

    if temperature_teacher <= 0:
        raise ValueError("temperature_teacher must be > 0!")

    def softmax(X, temp):
        X = (X / temp).exp()
        return X / (X.sum(-1, keepdims=True) + 1e-20)

    bs = teacher_logits.size(0)
    neg_K = teacher_logits.size(1) - 1

    teacher_logits_clamped = torch.clamp(teacher_logits, min=-20, max=20)
    teacher_soft_full = softmax(teacher_logits_clamped, temperature_teacher)

    teacher_soft = teacher_soft_full[..., 0]

    teacher_logits_weights = teacher_soft.clone()
    if teacher_logits_weights.size(1) > 1:
        teacher_logits_weights[:, 1:] = 1 - teacher_logits_weights[:, 1:]

    weighted_logits = student_logits * teacher_logits_weights
    softmax_weighted = softmax(weighted_logits, temperature)

    inputs = torch.clamp(softmax_weighted, min=1e-8).log()

    labels = torch.zeros(bs, dtype=torch.long, device=student_logits.device)

    loss_bs = nll_criterion(inputs, labels)
    loss_bs = loss_bs * loss_hardneg_weight

    return loss_bs.sum() / (bs * neg_K)


def cal_loss_rd(args, teacher_logits, student_logits, teacher_temperature):
    loss_pearson_weight = args.beta

    def softmax(X, temp):
        X = X / temp
        max_vals, _ = X.max(dim=-1, keepdim=True)
        X = X - max_vals
        X = X.exp()
        res = X / (X.sum(-1, keepdim=True) + 1e-20)
        return res

    def pearsonr(x, y, batch_first=True):
        assert x.shape == y.shape
        dim = -1 if batch_first else 0
        assert x.shape[dim] > 1

        centered_x = x - x.mean(dim=dim, keepdim=True)
        centered_y = y - y.mean(dim=dim, keepdim=True)

        covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
        bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

        x_std = x.std(dim=dim, keepdim=True)
        y_std = y.std(dim=dim, keepdim=True)

        denom = (x_std * y_std) + 1e-8
        corr = bessel_corrected_covariance / denom

        return corr

    bs = student_logits.size(0)

    teacher_logits = softmax(teacher_logits, teacher_temperature)[:, :, 0]

    spearson = pearsonr(student_logits, teacher_logits).squeeze()

    loss_bs = 1 - spearson

    loss_bs = loss_bs * loss_pearson_weight

    return loss_bs.sum() / bs


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
    sigmoid_diff = sigmoid(diff) + 1e-8
    log_sigmoid_diff = sigmoid_diff.log()

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

    loss_bs = (loss_hardneg_inbatch * weight_hardneg_inbatch).sum(-1)
    loss_bs = loss_bs * loss_bpr_weight

    return loss_bs.sum() / (bs * neg_K * inbatch)


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
