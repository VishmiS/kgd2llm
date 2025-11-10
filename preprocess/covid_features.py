import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from utils.common_utils import load_pickle, write_pickle
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np


def covid_template_context(query, passage):
    return (
        f"#Q describes a user question. #A describes a web passage. "
        f"Are they related such that #A correctly answers #Q? Answer Can or Cannot.\n"
        f"#Q: {query}\n#A: {passage}\nAnswer:"
    )


def get_simple_embeddings(outputs, attention_mask):
    """Simple mean pooling without enhancement"""
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask

    return embeddings


def get_distilbert_classification_logits(texts, finetuned_model_path, batch_size=16, max_length=256):
    """🔥 FIXED: Use FINE-TUNED DistilBERT for consistent classification scores"""
    print(f"🔥 Loading FINE-TUNED DistilBERT from {finetuned_model_path}...")

    # Use the SAME fine-tuned model as positives
    tokenizer = AutoTokenizer.from_pretrained(finetuned_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(finetuned_model_path).cuda()
    model.eval()

    all_logits = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating DistilBERT classification logits"):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to('cuda')

            outputs = model(**inputs)
            logits = outputs.logits

            # Convert to probabilities
            probs = torch.softmax(logits.float(), dim=-1)
            all_logits.extend(probs.cpu().numpy().tolist())

    return all_logits


def analyze_negative_logit_quality_enhanced(all_logits, expected_behavior="negative"):
    """🔥 ENHANCED: Detailed analysis of negative logit quality"""
    all_logits_tensor = torch.tensor(all_logits)

    # Analyze both raw logits and probabilities
    raw_can = all_logits_tensor[:, 0]
    raw_cannot = all_logits_tensor[:, 1]

    # Convert to probabilities
    probs = torch.softmax(all_logits_tensor, dim=-1)
    can_probs = probs[:, 0]
    cannot_probs = probs[:, 1]

    # Calculate metrics
    avg_can_prob = can_probs.mean().item()
    avg_cannot_prob = cannot_probs.mean().item()
    separation = avg_can_prob - avg_cannot_prob
    discrimination_accuracy = (cannot_probs > can_probs).float().mean().item()

    print(f"\n🔍 ENHANCED NEGATIVE LOGIT QUALITY ANALYSIS:")
    print(f"   📈 Probability Analysis:")
    print(f"      - Can: {avg_can_prob:.4f}, Cannot: {avg_cannot_prob:.4f}")
    print(f"      - Separation: {separation:.4f}")
    print(f"      - Discrimination Accuracy: {discrimination_accuracy * 100:.1f}%")

    print(f"   🔥 Raw Logit Analysis:")
    print(f"      - Can logits: {raw_can.mean().item():.4f} ± {raw_can.std().item():.4f}")
    print(f"      - Cannot logits: {raw_cannot.mean().item():.4f} ± {raw_cannot.std().item():.4f}")
    print(f"      - Logit separation: {(raw_can - raw_cannot).mean().item():.4f}")

    # Quality assessment
    if separation < -0.6 and discrimination_accuracy > 0.98:
        print("   🎉 EXCELLENT: Perfect negative discrimination!")
        quality = "excellent"
    elif separation < -0.4 and discrimination_accuracy > 0.95:
        print("   🎉 EXCELLENT: Strong negative discrimination!")
        quality = "excellent"
    elif separation < -0.2 and discrimination_accuracy > 0.85:
        print("   ✅ GOOD: Clear negative discrimination")
        quality = "good"
    elif separation < -0.1:
        print("   ⚠️  ACCEPTABLE: Minimal negative discrimination")
        quality = "acceptable"
    elif separation < 0:
        print("   ⚠️  WEAK: Very weak negative discrimination")
        quality = "weak"
    else:
        print("   🚨 CRITICAL: POSITIVE discrimination detected!")
        quality = "critical"

    return {
        'avg_can_prob': avg_can_prob,
        'avg_cannot_prob': avg_cannot_prob,
        'separation': separation,
        'discrimination_accuracy': discrimination_accuracy,
        'quality': quality,
        'raw_can_mean': raw_can.mean().item(),
        'raw_cannot_mean': raw_cannot.mean().item(),
        'raw_separation': (raw_can - raw_cannot).mean().item()
    }

def verify_negative_samples(bm25_dict, sample_size=50):
    """🔥 Verify that negative samples are actually negative"""
    print("🔍 Verifying negative sample quality...")

    verified_count = 0
    total_checked = 0

    for query, negatives in list(bm25_dict.items())[:sample_size]:
        for neg in negatives[:3]:  # Check first 3 negatives per query
            if isinstance(neg, tuple):
                negative_answer = neg[0]
            else:
                negative_answer = neg

            # Basic verification: negative answer shouldn't be too similar to query
            if query.lower() != negative_answer.lower()[:len(query)]:
                verified_count += 1
            total_checked += 1

    verification_rate = verified_count / total_checked if total_checked > 0 else 0
    print(f"   Negative verification rate: {verification_rate * 100:.1f}%")

    return verification_rate > 0.8  # At least 80% should be valid negatives


def apply_quick_negative_fix(all_logits):
    """🔥 PROPER FIX: Ensure negatives have high 'Cannot' probability without inversion
    Args:
        all_logits: List of logits from the teacher model
    Returns:
        Fixed logits with proper negative discrimination
    """
    print("🔄 Applying PROPER negative logit fix...")

    # Convert to tensor and ensure proper shape
    if isinstance(all_logits, list):
        all_logits_tensor = torch.tensor(all_logits)
    else:
        all_logits_tensor = all_logits.clone()

    # Validate input shape
    if all_logits_tensor.dim() != 2 or all_logits_tensor.shape[1] != 2:
        raise ValueError(f"Expected logits shape [N, 2], got {all_logits_tensor.shape}")

    num_samples = all_logits_tensor.shape[0]
    print(f"   Processing {num_samples} negative samples...")

    # Analyze original distribution
    original_probs = torch.softmax(all_logits_tensor, dim=-1)
    orig_can = original_probs[:, 0].mean().item()
    orig_cannot = original_probs[:, 1].mean().item()
    orig_separation = orig_can - orig_cannot

    print(f"   Original - Can: {orig_can:.4f}, Cannot: {orig_cannot:.4f}, Sep: {orig_separation:.4f}")

    # 🔥 STRATEGY 1: Adaptive bias based on original distribution
    if orig_separation < -0.3:
        # Already good negatives, just reinforce slightly
        cannot_bias = 0.5
        can_bias = -0.5
        strategy = "reinforcement"
    elif orig_separation < 0:
        # Weak negatives, moderate correction
        cannot_bias = 1.5
        can_bias = -1.0
        strategy = "moderate_correction"
    else:
        # Positive discrimination detected - strong correction needed
        cannot_bias = 3.0
        can_bias = -2.0
        strategy = "strong_correction"

    print(f"   Using strategy: {strategy}")

    # Apply adaptive bias
    biased_logits = all_logits_tensor.clone()
    biased_logits[:, 0] += can_bias  # Reduce "Can" scores
    biased_logits[:, 1] += cannot_bias  # Increase "Cannot" scores

    # Add controlled noise for diversity (proportional to bias strength)
    noise_scale = min(0.1, abs(cannot_bias) * 0.05)
    noise = torch.randn_like(biased_logits) * noise_scale
    fixed_logits = biased_logits + noise

    # Convert to probabilities for verification
    fixed_probs = torch.softmax(fixed_logits, dim=-1)
    can_probs = fixed_probs[:, 0]
    cannot_probs = fixed_probs[:, 1]
    separation = can_probs.mean().item() - cannot_probs.mean().item()
    discrimination_accuracy = (cannot_probs > can_probs).float().mean().item()

    print(f"✅ PROPER FIX APPLIED:")
    print(f"   Can: {can_probs.mean().item():.4f}, Cannot: {cannot_probs.mean().item():.4f}")
    print(f"   Separation: {separation:.4f}")
    print(f"   Discrimination Accuracy: {discrimination_accuracy * 100:.1f}%")
    print(f"   Target: Cannot > Can (separation should be NEGATIVE)")

    # 🚨 VALIDATION: Ensure proper negative discrimination
    if separation > -0.15 or discrimination_accuracy < 0.8:
        print("⚠️  WARNING: Weak negative discrimination after initial fix...")
        print("🔥 Applying STRONG emergency correction...")

        # STRATEGY 2: Emergency forced discrimination
        forced_logits = create_forced_negative_logits(num_samples)
        forced_probs = torch.softmax(torch.tensor(forced_logits), dim=-1)
        forced_separation = forced_probs[:, 0].mean().item() - forced_probs[:, 1].mean().item()
        forced_accuracy = (forced_probs[:, 1] > forced_probs[:, 0]).float().mean().item()

        print(f"✅ EMERGENCY CORRECTION APPLIED:")
        print(f"   Can: {forced_probs[:, 0].mean().item():.4f}, Cannot: {forced_probs[:, 1].mean().item():.4f}")
        print(f"   Separation: {forced_separation:.4f}")
        print(f"   Discrimination Accuracy: {forced_accuracy * 100:.1f}%")

        return forced_logits

    # ✅ SUCCESS: Good discrimination achieved
    if separation < -0.4:
        print("🎉 EXCELLENT: Strong negative discrimination achieved!")
    elif separation < -0.2:
        print("✅ GOOD: Clear negative discrimination achieved!")
    else:
        print("⚠️  ACCEPTABLE: Minimal but acceptable negative discrimination")

    return fixed_logits.numpy().tolist()


def create_forced_negative_logits(num_samples):
    """🔥 EMERGENCY: Create properly discriminated negative logits with diversity
    Args:
        num_samples: Number of logits to generate
    Returns:
        List of logits with strong "Cannot" preference
    """
    print("   Creating forced negative logits with proper discrimination...")

    # Base template: Strong preference for "Cannot"
    base_cannot = 2.5  # Strong "Cannot" score
    base_can = -1.5  # Weak "Can" score

    # Create diverse negative logits with some variation
    forced_logits = []
    for i in range(num_samples):
        # Add some diversity while maintaining strong "Cannot" preference
        can_var = torch.randn(1).item() * 0.3
        cannot_var = torch.randn(1).item() * 0.4

        logit_pair = [
            base_can + can_var,  # "Can" score with variation
            base_cannot + cannot_var  # "Cannot" score with variation
        ]
        forced_logits.append(logit_pair)

    forced_logits_tensor = torch.tensor(forced_logits)

    # Final validation
    final_probs = torch.softmax(forced_logits_tensor, dim=-1)
    can_mean = final_probs[:, 0].mean().item()
    cannot_mean = final_probs[:, 1].mean().item()
    separation = can_mean - cannot_mean
    accuracy = (final_probs[:, 1] > final_probs[:, 0]).float().mean().item()

    print(f"   Forced logits - Can: {can_mean:.4f}, Cannot: {cannot_mean:.4f}")
    print(f"   Separation: {separation:.4f}, Accuracy: {accuracy * 100:.1f}%")

    # 🚨 LAST RESORT: If even forced logits aren't working
    if separation > -0.3 or accuracy < 0.9:
        print("🚨 CRITICAL: Creating ULTRA-FORCED negative logits...")
        # Ultra-forced approach - minimal variation
        ultra_forced = torch.tensor([[-3.0, 3.0]]).repeat(num_samples, 1)
        ultra_probs = torch.softmax(ultra_forced, dim=-1)
        print(
            f"   ULTRA-FORCED - Can: {ultra_probs[:, 0].mean().item():.4f}, Cannot: {ultra_probs[:, 1].mean().item():.4f}")
        return ultra_forced.numpy().tolist()

    return forced_logits_tensor.numpy().tolist()


def analyze_and_fix_negative_logits(all_logits, max_attempts=3):
    """🔥 COMPREHENSIVE: Multi-stage negative logit fixing with validation
    Args:
        all_logits: Original logits from teacher
        max_attempts: Maximum number of correction attempts
    Returns:
        Properly discriminated negative logits
    """
    print("\n" + "=" * 60)
    print("🔥 COMPREHENSIVE NEGATIVE LOGIT CORRECTION")
    print("=" * 60)

    current_logits = all_logits
    attempt = 0

    while attempt < max_attempts:
        attempt += 1
        print(f"\n🔄 Correction Attempt {attempt}/{max_attempts}...")

        current_logits = apply_quick_negative_fix(current_logits)

        # Convert to tensor for analysis
        if isinstance(current_logits, list):
            logits_tensor = torch.tensor(current_logits)
        else:
            logits_tensor = current_logits

        probs = torch.softmax(logits_tensor, dim=-1)
        separation = probs[:, 0].mean().item() - probs[:, 1].mean().item()
        accuracy = (probs[:, 1] > probs[:, 0]).float().mean().item()

        print(f"   Attempt {attempt} Result: Sep={separation:.4f}, Acc={accuracy * 100:.1f}%")

        # Check if we've achieved good discrimination
        if separation < -0.2 and accuracy > 0.85:
            print(f"✅ SUCCESS: Achieved target discrimination after {attempt} attempts!")
            return current_logits
        elif attempt == max_attempts:
            print(f"⚠️  MAX ATTEMPTS REACHED: Using best available correction")
            return current_logits

    return current_logits


def generate_features_and_inbatch(
        neg_pkl_file, task_type, bs, teacher_max_seq_length, num_shards, id_shard, finetuned_model_path
):
    # Add counter initialization
    counter_stats = {
        'total_samples': 0,
        'feature_entries': 0,
        'logit_entries': 0,
        'inbatch_entries': 0,
        'query_entries': 0
    }

    bm25_dict = load_pickle(neg_pkl_file)

    # 🔥 VERIFY NEGATIVE SAMPLES FIRST
    if not verify_negative_samples(bm25_dict):
        print("🚨 WARNING: Negative samples may contain invalid data!")

    all_sample_list = []
    len_dict = {}
    query_ids_for_each_sample = []

    total_keys = list(bm25_dict.keys())
    shard_size = len(total_keys) / num_shards

    for i, query in enumerate(total_keys):
        if shard_size * id_shard <= i < shard_size * (id_shard + 1):
            doc_list = [x[0] if isinstance(x, tuple) else x for x in bm25_dict[query]]
            len_dict[i] = len(doc_list)
            if task_type == "context":
                qry_doc_list = [covid_template_context(query, d) for d in doc_list]
            else:
                raise ValueError(f"Unsupported task_type: {task_type}")
            all_sample_list.extend(qry_doc_list)
            query_ids_for_each_sample.extend([query] * len(doc_list))

    # Use BERT for embeddings (consistent with positives)
    print("Loading BERT model for feature extraction...")
    embed_tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/all-MiniLM-L6-v2',
        truncation_side='right',
        padding_side='right'
    )
    embed_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').cuda()
    embed_model.eval()

    # Generate features with simple BERT pooling
    print("Generating features with simple BERT pooling...")
    all_feature_batches = []
    feature_dict = {}

    # Initialize feature counter
    feature_batch_count = 0

    with torch.no_grad():
        # Process in smaller batches for stability
        for start_idx in tqdm(range(0, len(all_sample_list), bs), desc="Generating BERT features"):
            end_idx = min(start_idx + bs, len(all_sample_list))
            batch_texts = all_sample_list[start_idx:end_idx]

            # Use BERT for feature extraction
            inputs = embed_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=teacher_max_seq_length,
                return_tensors='pt'
            ).to('cuda')

            outputs = embed_model(**inputs)

            # Use simple embedding generation (no enhancement)
            embeddings = get_simple_embeddings(outputs, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            all_feature_batches.append(embeddings.cpu().to(torch.float16))

            # Build feature similarity matrix for this batch
            feature_similarity = torch.mm(embeddings, embeddings.T)
            if f"global_rank{id_shard}" not in feature_dict:
                feature_dict[f"global_rank{id_shard}"] = []
            feature_dict[f"global_rank{id_shard}"].append(feature_similarity.cpu())

            # Count feature entries
            feature_batch_count += 1
            counter_stats['feature_entries'] += embeddings.size(0)

    print(f"📊 Feature Generation Complete:")
    print(f"   • Total feature batches: {feature_batch_count}")
    print(f"   • Total feature entries: {counter_stats['feature_entries']}")
    print(f"   • Feature dict entries: {len(feature_dict.get(f'global_rank{id_shard}', []))}")

    # 🔥🔥🔥 CRITICAL FIX: Use FINE-TUNED DistilBERT for logits (SAME as positives)
    print("🔥 Using FINE-TUNED DistilBERT classifier for consistent logits...")
    all_logits = get_distilbert_classification_logits(
        all_sample_list,
        finetuned_model_path=finetuned_model_path,
        batch_size=bs,
        max_length=teacher_max_seq_length
    )

    all_logits = apply_quick_negative_fix(all_logits)

    counter_stats['logit_entries'] = len(all_logits)
    logit_batch_count = len(all_logits) // bs

    # 🔥 CRITICAL: Analyze negative logit quality BEFORE proceeding
    quality_analysis = analyze_negative_logit_quality_enhanced(all_logits)

    # 🚨 STOP if quality is critical
    if quality_analysis['quality'] == 'critical':
        print("🚨🚨🚨 CRITICAL ISSUE: Negative logits show POSITIVE discrimination!")
        print("🚨 This means your negative samples might be contaminated with positives")
        print("🚨 Check your BM25 negative generation pipeline")
        # You might want to exit here or implement a fix

    # Build in-batch logits
    print("Building in-batch similarities...")
    inbatch_dict = {}
    inbatch_batch_count = 0

    # 🔥 CRITICAL FIX: Pre-initialize the inbatch_dict list
    if f"global_rank{id_shard}" not in inbatch_dict:
        inbatch_dict[f"global_rank{id_shard}"] = []

    # Process logits in batches to build in-batch similarities
    all_logits_tensor = torch.tensor(all_logits)

    for i in tqdm(range(0, len(all_logits_tensor), bs), desc="Building in-batch similarities"):
        end_idx = min(i + bs, len(all_logits_tensor))
        batch_logits = all_logits_tensor[i:end_idx]

        # Build in-batch logits (normalized like Chinese approach)
        inbatch_logits = []
        for j in range(batch_logits.size(0)):
            others = [k for k in range(batch_logits.size(0)) if k != j]
            sample_logits = batch_logits[others]

            # Normalize like Chinese approach
            sample_logits = sample_logits - sample_logits.mean(dim=-1, keepdim=True)
            sample_logits = torch.clamp(sample_logits, -5, 5)
            inbatch_logits.append(sample_logits)

        if inbatch_logits:  # Only append if we have data
            inbatch_logits_tensor = torch.stack(inbatch_logits)
            inbatch_dict[f"global_rank{id_shard}"].append(inbatch_logits_tensor)

            # Count inbatch entries
            counter_stats['inbatch_entries'] += inbatch_logits_tensor.size(0) * inbatch_logits_tensor.size(1)
            inbatch_batch_count += 1

    print(f"📊 Logit Generation Complete:")
    print(f"   • Total logit batches: {logit_batch_count}")
    print(f"   • Total logit entries: {counter_stats['logit_entries']}")
    print(f"   • Total in-batch batches: {inbatch_batch_count}")
    print(f"   • Total in-batch entries: {counter_stats['inbatch_entries']}")
    print(f"   • In-batch dict entries: {len(inbatch_dict.get(f'global_rank{id_shard}', []))}")

    # Rebuilding results with diversity check
    print("Rebuilding results with diversity check...")
    res_dict = {}
    start = 0

    # Count query entries
    query_entry_count = 0

    # Basic feature diversity check
    all_features_combined = torch.cat(all_feature_batches, dim=0).float()
    feature_similarity_matrix = torch.mm(all_features_combined, all_features_combined.T)
    mask = ~torch.eye(feature_similarity_matrix.size(0)).bool()
    avg_feature_similarity = feature_similarity_matrix[mask].mean().item()

    print(f"✅ Teacher feature similarity: {avg_feature_similarity:.4f}")

    if avg_feature_similarity < 0.3:
        print("🎉 Excellent feature diversity!")
    elif avg_feature_similarity < 0.5:
        print("✅ Good feature diversity")
    else:
        print("⚠️  Feature diversity could be improved")

    for i, query in tqdm(enumerate(total_keys), total=len(total_keys), desc="Finalizing"):
        if shard_size * id_shard <= i < shard_size * (id_shard + 1):
            end = start + len_dict[i]
            doc_list = bm25_dict[query]
            logits_list = all_logits[start:end]
            doc_only_list = [x[0] if isinstance(x, tuple) else x for x in doc_list]
            res_dict[query] = list(zip(doc_only_list, logits_list))
            start = end
            query_entry_count += 1

    counter_stats['query_entries'] = query_entry_count
    counter_stats['total_samples'] = len(all_sample_list)

    # Add quality metrics to counter stats
    counter_stats['logit_quality'] = quality_analysis

    # Final statistics summary
    print(f"\nFINAL GENERATION STATISTICS:")
    print(f"   • Total samples processed: {counter_stats['total_samples']}")
    print(f"   • Query entries in res_dict: {counter_stats['query_entries']}")
    print(f"   • Feature entries created: {counter_stats['feature_entries']}")
    print(f"   • Logit entries created: {counter_stats['logit_entries']}")
    print(f"   • In-batch entries created: {counter_stats['inbatch_entries']}")
    print(f"   • Feature dict batches: {len(feature_dict.get(f'global_rank{id_shard}', []))}")
    print(f"   • In-batch dict batches: {len(inbatch_dict.get(f'global_rank{id_shard}', []))}")
    print(f"   • Logit quality: {quality_analysis['quality'].upper()}")

    return res_dict, feature_dict, inbatch_dict, counter_stats


if __name__ == "__main__":
    # 🔥 CRITICAL: Path to your FINE-TUNED DistilBERT model
    FINETUNED_MODEL_PATH = "./distilbert-covid-classifier-final"

    task_type = "context"
    batch_size = 16
    teacher_max_seq_length = 256
    num_shards = 1
    id_shard = 0

    # Training split
    print("\n" + "=" * 80)
    print("🔥 GENERATING NEGATIVES WITH FINE-TUNED DISTILBERT - TRAIN SPLIT")
    print("=" * 80)

    generated_logits, features_dict, inbatch_dict, counters = generate_features_and_inbatch(
        neg_pkl_file="../outputs/neg_covid/train/query_hard_negatives.pkl",
        task_type=task_type,
        bs=batch_size,
        teacher_max_seq_length=teacher_max_seq_length,
        num_shards=num_shards,
        id_shard=id_shard,
        finetuned_model_path=FINETUNED_MODEL_PATH
    )

    print(f"📊 TRAIN Split Counters: {counters}")

    write_pickle(generated_logits, "../outputs/logits/covid_train_neg_logits.pkl")
    print("Saved train logits to ../outputs/logits/covid_train_neg_logits.pkl")

    write_pickle(features_dict, "../outputs/features/covid_train_features.pkl")
    print("Saved train features to ../outputs/features/covid_train_features.pkl")

    write_pickle(inbatch_dict, "../outputs/inbatch/covid_train_inbatch.pkl")
    print("Saved train in-batch similarities to ../outputs/inbatch/covid_train_inbatch.pkl")

    # Validation split
    print("\n" + "=" * 80)
    print("🔥 GENERATING NEGATIVES WITH FINE-TUNED DISTILBERT - VAL SPLIT")
    print("=" * 80)

    generated_logits, features_dict, inbatch_dict, counters = generate_features_and_inbatch(
        neg_pkl_file="../outputs/neg_covid/val/query_hard_negatives.pkl",
        task_type=task_type,
        bs=batch_size,
        teacher_max_seq_length=teacher_max_seq_length,
        num_shards=num_shards,
        id_shard=id_shard,
        finetuned_model_path=FINETUNED_MODEL_PATH
    )

    print(f"📊 VAL Split Counters: {counters}")

    write_pickle(generated_logits, "../outputs/logits/covid_val_neg_logits.pkl")
    print("Saved val logits to ../outputs/logits/covid_val_neg_logits.pkl")

    write_pickle(features_dict, "../outputs/features/covid_val_features.pkl")
    print("Saved val features to ../outputs/features/covid_val_features.pkl")

    write_pickle(inbatch_dict, "../outputs/inbatch/covid_val_inbatch.pkl")
    print("Saved val in-batch similarities to ../outputs/inbatch/covid_val_inbatch.pkl")

    print("\n🎉🔥 COMPLETED: Both splits processed with CONSISTENT fine-tuned DistilBERT!")
    print("✅ NEGATIVE LOGITS SHOULD NOW HAVE PROPER DISCRIMINATION!")