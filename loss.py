import warnings
import logging
from utils.common_utils import *

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings('ignore')


def cal_loss_in_batch(args, student_logits, temperature, criterion):
    bs = student_logits.size(0)
    logits = student_logits / temperature
    labels = torch.arange(bs, device=logits.device)
    loss_bs = criterion(logits, labels)

    return (loss_bs.sum()) / (bs * bs)


def cal_loss_hardneg(args, teacher_logits, student_logits, temperature_teacher, temperature, nll_criterion):
    loss_hardneg_weight = args.alpha

    def softmax(X, temp):
        X = (X / temp).exp()
        res = X / (X.sum(-1, keepdims=True) + 1e-20)
        return res

    bs = teacher_logits.size(0)
    neg_K = teacher_logits.size(1) - 1
    teacher_logits = softmax(teacher_logits, temperature_teacher)[:, :, 0]
    teacher_logits[:, 1:] = 1 - teacher_logits[:, 1:]
    inputs = (softmax(student_logits * teacher_logits, temperature)).log()
    labels = torch.zeros(bs, dtype=torch.long, device=student_logits.device)
    loss_bs = nll_criterion(inputs, labels)

    loss_bs = loss_bs * loss_hardneg_weight
    return loss_bs.sum() / (bs * neg_K)


def cal_loss_rd(args, teacher_logits, student_logits, teacher_temperature):
    loss_pearson_weight = args.beta

    def softmax(X, temp):
        X = (X / temp).exp()
        res = X / (X.sum(-1, keepdims=True) + 1e-20)
        return res

    def pearsonr(x, y, batch_first=True):
        assert x.shape == y.shape
        if batch_first:
            dim = -1
        else:
            dim = 0
        assert x.shape[dim] > 1
        centered_x = x - x.mean(dim=dim, keepdim=True)
        centered_y = y - y.mean(dim=dim, keepdim=True)
        covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)
        bessel_corrected_covariance = covariance / (x.shape[dim] - 1)
        x_std = x.std(dim=dim, keepdim=True)
        y_std = y.std(dim=dim, keepdim=True)
        corr = bessel_corrected_covariance / ((x_std * y_std) + 1e-8)
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

    def softmax(X, temp):
        X = (X / temp).exp()
        res = X / (X.sum(-1, keepdims=True) + 1e-20)
        return res

    teacher_logits_hardneg = softmax(teacher_logits_pos_hardneg, teacher_temperature)[:, :, 0]
    teacher_logits_inbatch = softmax(teacher_logits_pos_inbatch, teacher_temperature)[:, :, 0]

    bs = student_logits_pos_hardneg.size(0)
    neg_K = student_logits_pos_hardneg.size(1) - 1
    inbatch = student_logits_pos_inbatch.size(1) - 1
    student_logits_hardneg = student_logits_pos_hardneg[:, 1:]
    eye = torch.eye(bs, dtype=torch.bool)
    student_logits_inbatch = student_logits_pos_inbatch[~eye].reshape(bs, -1)
    loss_hardneg_inbatch = -((sigmoid(student_logits_hardneg.view(bs, neg_K, 1).expand(-1, -1, inbatch).reshape(bs,
                                                                                                                -1) - student_logits_inbatch.unsqueeze(
        1).expand(-1, neg_K, -1).reshape(bs, -1)) + 1e-8).log())
    weight_hardneg_inbatch = teacher_logits_hardneg.repeat_interleave(inbatch, dim=1) - teacher_logits_inbatch.repeat(
        (1, neg_K))
    weight_hardneg_inbatch = torch.clamp(weight_hardneg_inbatch, min=0) / scale_param
    loss_bs = (loss_hardneg_inbatch * weight_hardneg_inbatch).sum(-1)
    loss_bs = loss_bs * loss_bpr_weight

    return loss_bs.sum() / (bs * neg_K * inbatch)

def cal_feat_loss(args, teacher_feat_cos, student_feature_pos_hardneg):
    loss_feat_weight = args.eta
    neg_K = teacher_feat_cos.size(1)
    student_feature_pos_hardneg = student_feature_pos_hardneg.transpose(0, 1)
    student_feature_pos_hardneg = student_feature_pos_hardneg / student_feature_pos_hardneg.norm(dim=-1, keepdim=True)
    student_feat_cos = torch.matmul(student_feature_pos_hardneg, student_feature_pos_hardneg.transpose(-2, -1))
    loss_bs = ((teacher_feat_cos - student_feat_cos) ** 2).sum((-1, -2))

    loss_bs = loss_bs * loss_feat_weight

    return loss_bs.sum() / (neg_K * neg_K)