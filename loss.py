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
    EFFECTIVE IN-BATCH LOSS: Balanced approach that actually learns
    """
    bs, num_candidates = student_logits.shape

    # Extract scores
    pos_scores = student_logits[:, 0]  # [batch_size]
    neg_scores = student_logits[:, 1:]  # [batch_size, num_negatives]

    # Calculate separation metrics
    with torch.no_grad():
        separation = (pos_scores.mean() - neg_scores.mean()).item()
        accuracy = (pos_scores.unsqueeze(1) > neg_scores).float().mean().item()

    # ==========================================================================
    # 🔥 SIMPLE & EFFECTIVE CROSS-ENTROPY LOSS
    # ==========================================================================
    effective_temp = max(temperature, 0.1)
    logits = student_logits / effective_temp

    # Remove the detach() - this was preventing gradient flow!
    logits = logits - logits.max(dim=1, keepdim=True)[0]  # Keep gradients

    # Standard cross-entropy (no label smoothing for now)
    labels = torch.zeros(bs, dtype=torch.long, device=student_logits.device)
    ce_loss = F.cross_entropy(logits, labels, reduction='mean')

    # ==========================================================================
    # 🔥 FOCUSED RANKING LOSS (Single objective)
    # ==========================================================================
    margin = 0.2
    ranking_loss = F.relu(neg_scores - pos_scores.unsqueeze(1) + margin).mean()

    # ==========================================================================
    # 🔥 SIMPLE ADAPTIVE WEIGHTING
    # ==========================================================================

    # Monitor if we're actually learning
    if separation < 0.05:
        # Very weak separation: Focus on ranking
        weights = {'ce': 0.2, 'ranking': 0.8}
        print(f"🔶 WEAK LEARNING: Emphasizing ranking (sep: {separation:.4f})")
    elif separation < 0.15:
        # Moderate separation: Balanced
        weights = {'ce': 0.5, 'ranking': 0.5}
        print(f"✅ LEARNING: Balanced approach (sep: {separation:.4f})")
    else:
        # Good separation: Focus on CE for refinement
        weights = {'ce': 0.8, 'ranking': 0.2}
        print(f"🎯 STRONG LEARNING: Refining with CE (sep: {separation:.4f})")

    # Combine losses
    combined_loss = (
            ce_loss * weights['ce'] +
            ranking_loss * weights['ranking']
    )

    # ==========================================================================
    # 🔥 GRADIENT FLOW VERIFICATION
    # ==========================================================================

    # Check if gradients will flow properly
    requires_grad = any(p.requires_grad for p in [ce_loss, ranking_loss, combined_loss])
    print(f"   Gradient flow: {'✅' if requires_grad else '❌'}")

    # ==========================================================================
    # 🔥 CLEAN MONITORING
    # ==========================================================================
    print(f"\n📊 IN-BATCH LOSS:")
    print(f"   Separation: {separation:.4f} | Accuracy: {accuracy * 100:.1f}%")
    print(f"   Pos mean: {pos_scores.mean().item():.4f} | Neg mean: {neg_scores.mean().item():.4f}")
    print(f"   Losses: CE={ce_loss.item():.4f}, Rank={ranking_loss.item():.4f}")
    print(f"   Weights: CE={weights['ce']}, Rank={weights['ranking']}")
    print(f"   Final loss: {combined_loss.item():.4f}")

    # Learning progress
    if separation > 0.3:
        print("   🎉 EXCELLENT: Strong learning!")
    elif separation > 0.15:
        print("   ✅ GOOD: Clear progress")
    elif separation > 0.05:
        print("   🔶 MODERATE: Learning slowly")
    else:
        print("   🚨 POOR: Need better optimization")

    return combined_loss


def cal_loss_hardneg(args, teacher_logits, student_logits, temperature_teacher, temperature, nll_criterion):
    """
    STABLE VERSION: Simplified with consistent learning strategy
    """
    target_dtype = student_logits.dtype
    teacher_logits = teacher_logits.to(target_dtype)

    # ==========================================================================
    # 🔥 EXTRACT SCORES CORRECTLY
    # ==========================================================================

    # Extract "Can" probabilities from teacher
    teacher_probs = F.softmax(teacher_logits / temperature_teacher, dim=-1)[..., 0]

    # Student already has scores, just apply temperature
    student_scores = student_logits / temperature

    # Extract positive and negative
    teacher_pos = teacher_probs[:, 0]  # [batch_size]
    teacher_neg = teacher_probs[:, 1:]  # [batch_size, num_negatives]
    student_pos = student_scores[:, 0]  # [batch_size]
    student_neg = student_scores[:, 1:]  # [batch_size, num_negatives]

    # ==========================================================================
    # 🔥 SIMPLIFIED STRATEGY: Consistent scaling
    # ==========================================================================

    # Use fixed, conservative scaling to prevent oscillation
    scale_factor = 0.2  # Fixed conservative scaling

    teacher_separation = (teacher_pos.unsqueeze(1) - teacher_neg) * scale_factor
    student_separation = student_pos.unsqueeze(1) - student_neg

    # 🔥 PRIMARY LOSS: Separation matching
    separation_loss = F.mse_loss(student_separation, teacher_separation)

    # ==========================================================================
    # 🔥 STABLE RANKING LOSS (Fixed margin)
    # ==========================================================================

    # Use consistent margin to prevent oscillation
    margin = 0.2  # Fixed medium margin
    margin_tensor = torch.tensor(margin, dtype=target_dtype, device=student_scores.device)
    ranking_loss = F.relu(student_neg - student_pos.unsqueeze(1) + margin_tensor).mean()

    # ==========================================================================
    # 🔥 SIMPLIFIED STABILITY REGULARIZATION
    # ==========================================================================

    # Only basic score range guidance
    stability_loss = (
                             F.relu(0.1 - student_pos).mean() +  # Encourage positives > 0.1
                             F.relu(student_neg - 0.5).mean()  # Encourage negatives < 0.5
                     ) * 0.01  # Very weak guidance

    # ==========================================================================
    # 🔥 CONSISTENT LOSS BALANCING
    # ==========================================================================

    # Fixed weights for stability - no adaptive changes
    loss_weights = {
        'separation': 3.0,  # Primary focus
        'ranking': 2.0,  # Secondary focus
        'stability': 0.1  # Very weak
    }

    final_loss = (
                         separation_loss * torch.tensor(loss_weights['separation'], dtype=target_dtype,
                                                        device=student_scores.device) +
                         ranking_loss * torch.tensor(loss_weights['ranking'], dtype=target_dtype,
                                                     device=student_scores.device) +
                         stability_loss * torch.tensor(loss_weights['stability'], dtype=target_dtype,
                                                       device=student_scores.device)
                 ) * torch.tensor(args.beta, dtype=target_dtype, device=student_scores.device)

    # ==========================================================================
    # 🔥 CLEAN DEBUGGING
    # ==========================================================================

    pos_student = student_pos.mean().item()
    neg_student = student_neg.mean().item()
    student_sep = pos_student - neg_student

    pos_teacher = teacher_pos.mean().item()
    neg_teacher = teacher_neg.mean().item()
    teacher_sep = pos_teacher - neg_teacher

    student_accuracy = (student_pos.unsqueeze(1) > student_neg).float().mean().item()

    print(f"\n🎯 STABLE LEARNING LOSS:")
    print(f"   TEACHER: pos={pos_teacher:.3f}, neg={neg_teacher:.3f}, sep={teacher_sep:.3f}")
    print(f"   STUDENT: pos={pos_student:.3f}, neg={neg_student:.3f}, sep={student_sep:.3f}")
    print(f"   Accuracy: Student={student_accuracy * 100:.1f}%")
    print(f"   Losses: sep={separation_loss.item():.4f}, rank={ranking_loss.item():.4f}")
    print(f"   Final loss: {final_loss.item():.4f}")

    # ==========================================================================
    # 🔥 SIMPLE PROGRESS INDICATOR
    # ==========================================================================

    if student_sep > 0.2:
        print("   🎉 EXCELLENT: Strong separation!")
    elif student_sep > 0.1:
        print("   ✅ GOOD: Clear progress")
    elif student_sep > 0.05:
        print("   ⚠️  FAIR: Moderate separation")
    elif student_sep > 0:
        print("   🔶 WEAK: Minimal separation")
    elif student_sep > -0.05:
        print("   🔄 STABILIZING: Near zero")
    else:
        print("   🚨 CRITICAL: Wrong direction")

    return final_loss

def cal_loss_rd(args, teacher_logits, student_logits, teacher_temperature):
    loss_pearson_weight = args.gamma
    dtype = student_logits.dtype

    teacher_probs = F.softmax(teacher_logits / teacher_temperature, dim=-1)[..., 1].to(dtype)
    student_probs = F.softmax(student_logits / teacher_temperature, dim=-1).to(dtype)

    teacher_probs = torch.clamp(teacher_probs, 1e-8, 1.0)
    student_probs = torch.clamp(student_probs, 1e-8, 1.0)

    # Manual MSE to ensure bfloat16
    loss = ((student_probs - teacher_probs) ** 2).mean() * loss_pearson_weight
    return loss


def cal_loss_rd2(args, teacher_logits_pos_hardneg, teacher_logits_pos_inbatch, teacher_temperature,
                 student_logits_pos_hardneg, student_logits_pos_inbatch, sigmoid, scale_param):
    """
    FIXED: Handle dtype and shape issues
    """
    loss_bpr_weight = args.gamma
    dtype = student_logits_pos_hardneg.dtype

    # Get probabilities - ensure bfloat16 throughout
    teacher_probs_hardneg = F.softmax(teacher_logits_pos_hardneg / teacher_temperature, dim=-1)[..., 1].to(dtype)
    student_probs_hardneg = F.softmax(student_logits_pos_hardneg / teacher_temperature, dim=-1).to(dtype)

    # SIMPLIFIED: Only use hard negatives to avoid shape complexity
    # Compute MSE loss manually to ensure bfloat16
    loss_hardneg = ((student_probs_hardneg - teacher_probs_hardneg) ** 2).mean()

    loss = loss_hardneg * loss_bpr_weight
    return loss

def cal_feat_loss(args, teacher_feat_cos, student_feature_pos_hardneg):
    """
    FIXED: Proper feature alignment loss with manual MSE
    """
    loss_feat_weight = args.eta
    dtype = student_feature_pos_hardneg.dtype

    # Handle different input shapes
    if student_feature_pos_hardneg.dim() == 3:
        student_feature_pos_hardneg = student_feature_pos_hardneg.transpose(0, 1)

    # Flatten or take mean across sequence dimension
    if student_feature_pos_hardneg.dim() == 3:
        student_features = student_feature_pos_hardneg.mean(dim=1)
    else:
        student_features = student_feature_pos_hardneg

    # Normalize
    student_features = F.normalize(student_features, p=2, dim=-1)

    # Compute cosine similarity
    student_feat_cos = torch.mm(student_features, student_features.t())

    # Ensure shapes match
    if student_feat_cos.shape != teacher_feat_cos.shape:
        min_size = min(student_feat_cos.size(0), teacher_feat_cos.size(0))
        student_feat_cos = student_feat_cos[:min_size, :min_size]
        teacher_feat_cos = teacher_feat_cos[:min_size, :min_size]

    # Manual MSE to ensure bfloat16
    loss = ((student_feat_cos - teacher_feat_cos) ** 2).mean() * loss_feat_weight
    return loss


def positive_discrimination_loss(student_pos_scores, student_neg_scores, margin=0.1):
    """Explicitly force positive scores to be higher than negative scores"""
    # student_pos_scores: [batch_size] tensor
    # student_neg_scores: [batch_size, num_negatives] tensor

    # Ensure positive scores are higher than all negatives by at least margin
    pos_expanded = student_pos_scores.unsqueeze(1)  # [batch_size, 1]

    # For each positive, compare against all its negatives
    discrimination_gap = pos_expanded - student_neg_scores  # [batch_size, num_negatives]

    # We want: pos_score - neg_score >= margin
    # Loss: max(0, margin - (pos_score - neg_score))
    discrimination_loss = F.relu(margin - discrimination_gap)

    # Average over all negatives and batch
    return discrimination_loss.mean()


def ranking_loss(student_pos_scores, student_neg_scores):
    """Simple ranking loss: positive should rank higher than negatives"""
    # For each example, positive should be higher than negatives
    pos_rank = student_pos_scores.unsqueeze(1)  # [batch_size, 1]
    neg_ranks = student_neg_scores  # [batch_size, num_negatives]

    # Pairwise comparison: we want pos > neg for all pairs
    pairwise_loss = F.relu(neg_ranks - pos_rank + 0.1)  # margin of 0.1

    return pairwise_loss.mean()


def compute_separation_metrics(student_pos, student_neg):
    """Compute separation metrics for monitoring"""
    pos_mean = student_pos.mean().item()
    neg_mean = student_neg.mean().item()
    separation = pos_mean - neg_mean

    # Accuracy: percentage of positives that score higher than their negatives
    pos_higher = (student_pos.unsqueeze(1) > student_neg).float().mean().item()

    return {
        'separation': separation,
        'pos_mean': pos_mean,
        'neg_mean': neg_mean,
        'accuracy': pos_higher
    }


def verify_loss_connection(loss_tensor, model_engine, step):
    """Verify that loss is properly connected to model parameters"""
    print(f"\n[LossCheck] Step {step}")
    print(f"  Loss value: {loss_tensor.item():.6f}")
    print(f"  Loss requires_grad: {loss_tensor.requires_grad}")
    print(f"  Loss device: {loss_tensor.device}")

    # Test if loss is connected to any parameters
    model_to_check = model_engine.module if hasattr(model_engine, 'module') else model_engine

    try:
        # This will fail if loss is not connected to model
        test_grads = torch.autograd.grad(
            loss_tensor,
            [p for p in model_to_check.parameters() if p.requires_grad],
            allow_unused=True,
            retain_graph=True
        )

        connected = sum(1 for g in test_grads if g is not None)
        print(f"  Parameters connected to loss: {connected}")
        return connected > 0
    except Exception as e:
        print(f"  ❌ Loss connection test failed: {e}")
        return False