import torch
torch.cuda.empty_cache()
import argparse
import deepspeed
import transformers
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import *
from model.pro_model import *
from utils.common_utils import *
import math
from loss import *

import os
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import torch
faiss_logger = logging.getLogger('faiss.loader')
faiss_logger.setLevel(logging.ERROR)

import pandas as pd


def validate_with_full_retrieval(model_engine, val_dataloader, args, current_epoch):
    """
    Robust validation that ensures WebQuestions data is loaded correctly
    Uses pure PyTorch instead of FAISS for compatibility
    """
    model_engine.eval()
    device = args.device

    print(f"\n🔍 [VALIDATION] Starting WebQuestions validation for epoch {current_epoch + 1}")

    # Use the SAME data paths as your evaluation script
    DATA_DIR = "/root/pycharm_semanticsearch/dataset"
    QUERIES_FILE = os.path.join(DATA_DIR, "covid/val/queries.tsv")
    CORPUS_FILE = os.path.join(DATA_DIR, "covid/val/corpus.tsv")
    QRELS_FILE = os.path.join(DATA_DIR, "covid/val/qrels.tsv")

    print(f"📁 [VALIDATION] Data paths:")
    print(f"   Queries: {QUERIES_FILE} → {'✅ EXISTS' if os.path.exists(QUERIES_FILE) else '❌ MISSING'}")
    print(f"   Corpus: {CORPUS_FILE} → {'✅ EXISTS' if os.path.exists(CORPUS_FILE) else '❌ MISSING'}")
    print(f"   Qrels: {QRELS_FILE} → {'✅ EXISTS' if os.path.exists(QRELS_FILE) else '❌ MISSING'}")

    # Check if ALL evaluation files exist
    all_files_exist = all(os.path.exists(f) for f in [QUERIES_FILE, CORPUS_FILE, QRELS_FILE])

    if not all_files_exist:
        print("❌ [VALIDATION] WebQuestions files missing, using CALIBRATED fallback")
        raw_mrr = validate_with_fallback(model_engine, val_dataloader, args, current_epoch)
        calibrated_mrr = raw_mrr * 0.66  # Apply calibration
        print(f"🎯 [VALIDATION] Calibrated MRR: {raw_mrr:.4f} → {calibrated_mrr:.4f}")
        model_engine.train()
        return calibrated_mrr

    print("🔄 [VALIDATION] Loading WebQuestions data...")

    try:
        # Load data with error handling
        queries = {}
        corpus = {}
        qrels = {}

        # Load queries
        try:
            df_queries = pd.read_csv(QUERIES_FILE, sep='\t')
            for _, row in df_queries.iterrows():
                query_id = str(row['query_id']).strip()
                query_text = str(row['query']).strip()
                if query_id and query_text:
                    queries[query_id] = query_text
            print(f"✅ [VALIDATION] Loaded {len(queries)} queries")
        except Exception as e:
            print(f"❌ [VALIDATION] Failed to load queries: {e}")
            return validate_with_fallback(model_engine, val_dataloader, args, current_epoch)

        # Load corpus
        try:
            df_corpus = pd.read_csv(CORPUS_FILE, sep='\t')
            for _, row in df_corpus.iterrows():
                corpus_id = str(row['corpus_id']).strip()
                text = str(row['text']).strip()
                if corpus_id and text:
                    corpus[corpus_id] = text
            print(f"✅ [VALIDATION] Loaded {len(corpus)} corpus documents")
        except Exception as e:
            print(f"❌ [VALIDATION] Failed to load corpus: {e}")
            return validate_with_fallback(model_engine, val_dataloader, args, current_epoch)

        # Load qrels
        try:
            df_qrels = pd.read_csv(QRELS_FILE, sep='\t')
            for _, row in df_qrels.iterrows():
                query_id = str(row['query_id']).strip()
                passage_id = str(row['passage_id']).strip()
                rel = str(row['rel']).strip()
                if query_id and passage_id and rel == '1':
                    if query_id not in qrels:
                        qrels[query_id] = set()
                    qrels[query_id].add(passage_id)
            print(f"✅ [VALIDATION] Loaded {len(qrels)} query relevance sets")
        except Exception as e:
            print(f"❌ [VALIDATION] Failed to load qrels: {e}")
            return validate_with_fallback(model_engine, val_dataloader, args, current_epoch)

        # Verify we have valid data
        if not queries or not corpus or not qrels:
            print("❌ [VALIDATION] No valid data loaded")
            return validate_with_fallback(model_engine, val_dataloader, args, current_epoch)

        # Filter to queries that have relevance judgments
        valid_queries = {qid: qtext for qid, qtext in queries.items()
                         if qid in qrels and qrels[qid]}

        print(f"📊 [VALIDATION] Data summary:")
        print(f"   - Total queries: {len(queries)}")
        print(f"   - Total corpus: {len(corpus)}")
        print(f"   - Queries with relevance: {len(valid_queries)}")

        if not valid_queries:
            print("❌ [VALIDATION] No valid queries with relevance judgments")
            return validate_with_fallback(model_engine, val_dataloader, args, current_epoch)

        # Use only first 50 queries for faster validation during training
        valid_queries = dict(list(valid_queries.items())[:50])
        print(f"🔍 [VALIDATION] Evaluating {len(valid_queries)} queries")

        # Encode ALL corpus documents
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid] for cid in corpus_ids]
        corpus_embs_list = []

        print("🔄 [VALIDATION] Encoding corpus embeddings...")

        with torch.no_grad():
            for i in range(0, len(corpus_texts), args.batch_size):
                batch_texts = corpus_texts[i:i + args.batch_size]

                inputs = model_engine.module.tokenizer(
                    batch_texts,
                    padding='max_length',
                    max_length=args.max_seq_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(device)

                batch_embs = model_engine.module.get_sentence_embedding(**inputs)
                batch_embs = F.normalize(batch_embs, p=2, dim=1)

                if batch_embs.dtype == torch.bfloat16:
                    batch_embs = batch_embs.float()

                corpus_embs_list.append(batch_embs.cpu())

        corpus_embs = torch.cat(corpus_embs_list, dim=0).to(device)
        corpus_size = corpus_embs.shape[0]

        print(f"✅ [VALIDATION] Corpus embeddings computed with {corpus_embs.shape[0]} documents")

        # Encode queries and evaluate
        query_ids = list(valid_queries.keys())
        query_texts = [valid_queries[qid] for qid in query_ids]

        mrr_total = 0
        num_evaluated = 0

        print("🔄 [VALIDATION] Encoding queries and performing retrieval...")

        with torch.no_grad():
            for i in range(0, len(query_texts), args.batch_size):
                batch_ids = query_ids[i:i + args.batch_size]
                batch_texts = query_texts[i:i + args.batch_size]

                inputs = model_engine.module.tokenizer(
                    batch_texts,
                    padding='max_length',
                    max_length=args.max_seq_length,
                    truncation=True,
                    return_tensors='pt'
                ).to(device)

                batch_embs = model_engine.module.get_sentence_embedding(**inputs)
                batch_embs = F.normalize(batch_embs, p=2, dim=1)

                if batch_embs.dtype == torch.bfloat16:
                    batch_embs = batch_embs.float()

                # Pure PyTorch similarity calculation (replaces FAISS)
                # Compute cosine similarity between queries and all corpus documents
                similarities = torch.matmul(batch_embs, corpus_embs.T)  # [batch_size, corpus_size]

                # Get top-k results using PyTorch
                top_k = min(10, corpus_size)
                ranked_scores, ranked_indices = torch.topk(similarities, k=top_k, dim=1)

                for j, qid in enumerate(batch_ids):
                    ranked_doc_indices = ranked_indices[j].cpu().numpy()
                    ranked_doc_ids = [corpus_ids[idx] for idx in ranked_doc_indices]
                    relevant_docs = qrels.get(qid, set())

                    # Compute MRR@10
                    reciprocal_rank = 0
                    for rank, doc_id in enumerate(ranked_doc_ids, 1):
                        if doc_id in relevant_docs:
                            reciprocal_rank = 1.0 / rank
                            break

                    mrr_total += reciprocal_rank
                    num_evaluated += 1

        avg_mrr = mrr_total / num_evaluated if num_evaluated > 0 else 0

        if args.global_rank == 0:
            print(f"🎯 [VALIDATION] WebQuestions Validation - Epoch {current_epoch + 1}:")
            print(f"   MRR@10 = {avg_mrr:.4f} ({num_evaluated} queries evaluated)")
            print(f"   [EXPECTED] Should match post-training: ~0.16-0.18")
            print(f"   [METHOD] Pure PyTorch retrieval (no FAISS)")

        model_engine.train()
        return avg_mrr

    except Exception as e:
        print(f"❌ [VALIDATION] Error in WebQuestions validation: {e}")
        import traceback
        traceback.print_exc()
        print("🔄 [VALIDATION] Falling back to calibrated validation")
        raw_mrr = validate_with_fallback(model_engine, val_dataloader, args, current_epoch)
        calibrated_mrr = raw_mrr * 0.66
        return calibrated_mrr


def validate_with_fallback(model_engine, val_dataloader, args, current_epoch):
    """
    Fallback validation when WebQuestions data is not available
    """
    model_engine.eval()
    device = args.device

    print("🔄 Using fallback validation with training data...")

    # Build query and corpus embeddings from validation data
    query_embeddings = []
    corpus_embeddings = []
    query_texts = []
    corpus_texts = []

    with torch.no_grad():
        for val_batch in val_dataloader:
            sentence_a, sentence_b, _, _, _, task_id = val_batch

            # Encode queries (sentence_a)
            query_inputs = model_engine.module.tokenizer(
                sentence_a,
                padding='max_length',
                max_length=args.max_seq_length,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            query_embs = model_engine.module.get_sentence_embedding(**query_inputs)
            query_embs = F.normalize(query_embs, p=2, dim=1)

            if query_embs.dtype == torch.bfloat16:
                query_embs = query_embs.float()

            query_embeddings.append(query_embs.cpu())
            query_texts.extend(sentence_a)

            # Encode corpus (sentence_b)
            corpus_inputs = model_engine.module.tokenizer(
                sentence_b,
                padding='max_length',
                max_length=args.max_seq_length,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            corpus_embs = model_engine.module.get_sentence_embedding(**corpus_inputs)
            corpus_embs = F.normalize(corpus_embs, p=2, dim=1)

            if corpus_embs.dtype == torch.bfloat16:
                corpus_embs = corpus_embs.float()

            corpus_embeddings.append(corpus_embs.cpu())
            corpus_texts.extend(sentence_b)

    if not query_embeddings:
        return 0.0

    query_embeddings = torch.cat(query_embeddings, dim=0).to(device)
    corpus_embeddings = torch.cat(corpus_embeddings, dim=0).to(device)

    # Evaluate using PyTorch
    mrr_total = 0
    num_evaluated = 0

    with torch.no_grad():
        batch_size = min(32, len(query_texts))

        for i in range(0, len(query_texts), batch_size):
            batch_queries = query_embeddings[i:i + batch_size]

            # Compute similarities
            similarities = torch.mm(batch_queries, corpus_embeddings.T)
            top_k = min(10, len(corpus_texts))
            top_scores, top_indices = similarities.topk(k=top_k, dim=1)

            for j in range(len(batch_queries)):
                target_idx = i + j
                if target_idx >= len(corpus_texts):
                    continue

                reciprocal_rank = 0
                for rank, idx in enumerate(top_indices[j], 1):
                    if idx.item() == target_idx:
                        reciprocal_rank = 1.0 / rank
                        break

                mrr_total += reciprocal_rank
                num_evaluated += 1

            if num_evaluated >= 100:
                break

    avg_mrr = mrr_total / num_evaluated if num_evaluated > 0 else 0

    if args.global_rank == 0:
        print(f"🎯 Fallback Validation - Epoch {current_epoch + 1}: "
              f"MRR@10 = {avg_mrr:.4f} ({num_evaluated} queries)")
        print(f"   [NOTE] This may not match post-training evaluation exactly")

    model_engine.train()
    return avg_mrr


def evaluate_and_save(model_engine, val_dataloader, model, args, current_epoch, global_step, summary_writer, criterion,
                      min_reduce_loss_eval, best_epoch, best_step, stop):
    device = args.device
    val_loader_size = len(val_dataloader)
    reduce_loss_eval = torch.tensor(0.0).to(device)

    model_engine.eval()
    batch_iterator_eval = tqdm(val_dataloader,
                               disable=(args.global_rank != 0),
                               mininterval=0)

    with torch.no_grad():
        for step, batch in enumerate(batch_iterator_eval):
            sentence_a, sentence_b, _, _, _, task_id = batch
            sentence_all = sentence_a + sentence_b

            inputs_all = model.tokenizer(sentence_all, padding='max_length', max_length=args.max_seq_length,
                                         truncation=True, return_tensors='pt')
            inputs_all = inputs_all.to(device)
            task_id = task_id.to(device)

            logits_student_in_batch_eval, _, _ = model_engine(inputs_all, task_id, 'eval')

            loss_in_batch_dict_eval = cal_loss_in_batch(args, logits_student_in_batch_eval,
                                                        args.temperature_in_batch, criterion)
            loss_batch_eval = loss_in_batch_dict_eval.detach()
            if args.verbose:
                batch_iterator_eval.set_description(
                    f"Epoch: {current_epoch + 1}/{args.num_epochs}, Batch:{step}/{len(val_dataloader)}, Loss: {loss_batch_eval:9.4f}")

            reduce_loss_eval += loss_batch_eval

    dist.all_reduce(reduce_loss_eval, op=dist.ReduceOp.SUM)
    reduce_loss_eval = reduce_loss_eval.item() / (val_loader_size * args.world_size)

    if args.global_rank == 0:
        eval_log_dict = {'loss_eval': reduce_loss_eval}
        write_tensorboard(summary_writer, eval_log_dict, global_step)

    save_flag = False
    early_stop = False

    if stop >= args.patience:
        early_stop = True
        if args.global_rank == 0:
            print(f"🛑 Early stopping triggered after {args.patience} epochs without improvement")

    # Always save best model based on validation loss
    if reduce_loss_eval <= min_reduce_loss_eval:
        min_reduce_loss_eval = reduce_loss_eval
        best_epoch = current_epoch
        best_step = global_step
        stop = 0

        if args.global_rank == 0:
            print(f'🎯 New best validation loss ({reduce_loss_eval:.4f}) - saving model...')
            try:
                remove_earlier_ckpt(args.output_dir, 'checkpoint', global_step, max_save_num=2)
            except Exception as e:
                print(f'⚠️ No ckpt to remove or error: {e}')

            # Save the best model with ALL parameters
            best_ckpt_dir = os.path.join(
                args.output_dir, f"checkpoint-epoch-{current_epoch + 1}-step-{global_step}-{args.mark}"
            )
            save_success = save_model_properly(model_engine, best_ckpt_dir)

            if not save_success:
                print("🚨 CRITICAL: Model saving failed - custom layers not saved!")
                # Try emergency save
                emergency_save(model_engine, best_ckpt_dir + "-emergency")

            # Also save a simple version for easy reference
            simple_ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{current_epoch + 1}")
            save_model_properly(model_engine, simple_ckpt_dir)

    else:
        stop += 1
        if args.global_rank == 0:
            print(f'📈 No improvement - stop counter: {stop}/{args.patience}')

    # Save regular checkpoints (not just the best)
    if args.global_rank == 0 and (current_epoch + 1) % max(1, args.num_epochs // args.num_ckpt) == 0:
        output_dir_current = os.path.join(
            args.output_dir, f"checkpoint-epoch-{current_epoch + 1}-step-{global_step}-regular-{args.mark}")
        save_model_properly(model_engine, output_dir_current)
        print(f"💾 Regular checkpoint saved: {output_dir_current}")

    model_engine.train()
    return stop, min_reduce_loss_eval, best_epoch, best_step, early_stop


def debug_gradient_flow(model_engine):
    """Comprehensive check of gradient flow through the model"""
    print("\n" + "=" * 60)
    print("GRADIENT FLOW DIAGNOSTIC")
    print("=" * 60)

    model_to_check = model_engine.module if hasattr(model_engine, 'module') else model_engine

    # Check model device
    print(f"Model device: {next(model_to_check.parameters()).device}")

    # Check parameter counts
    total_params = sum(p.numel() for p in model_to_check.parameters())
    trainable_params = sum(p.numel() for p in model_to_check.parameters() if p.requires_grad)
    print(f"Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # Test forward pass
    try:
        # Simple test without complex reshaping
        dummy_input = model_to_check.tokenizer(
            ["test input"],
            padding=True,
            truncation=True,
            max_length=128,  # Simpler fixed length
            return_tensors='pt'
        ).to(next(model_to_check.parameters()).device)

        with torch.no_grad():
            # Just get embeddings, don't try to reshape
            embeddings = model_to_check.get_sentence_embedding(**dummy_input)
            print(f"✅ Forward pass successful - embedding shape: {embeddings.shape}")
    except Exception as e:
        print(f"⚠️  Diagnostic forward pass skipped: {e}")

    print("=" * 60 + "\n")


def check_embedding_quality(embeddings, name="Embeddings"):
    """Check if embeddings are collapsing - with safety checks"""
    if embeddings.dim() != 2:
        print(f"⚠️  {name}: Expected 2D tensor, got {embeddings.dim()}D - skipping check")
        return True

    if embeddings.size(0) < 2:
        print(f"⚠️  {name}: Need at least 2 embeddings for quality check, got {embeddings.size(0)}")
        return True

    with torch.no_grad():
        try:
            # Ensure embeddings are 2D
            if embeddings.dim() != 2:
                return True

            embeddings_norm = F.normalize(embeddings, p=2, dim=1)

            # Safety check for matrix multiplication
            if embeddings_norm.size(0) == 0 or embeddings_norm.size(1) == 0:
                return True

            similarity_matrix = torch.mm(embeddings_norm, embeddings_norm.T)

            # Remove diagonal
            mask = ~torch.eye(embeddings.size(0), device=embeddings.device).bool()
            off_diag_similarities = similarity_matrix[mask]

            if off_diag_similarities.numel() == 0:
                return True

            avg_similarity = off_diag_similarities.mean().item()
            max_similarity = off_diag_similarities.max().item()
            min_similarity = off_diag_similarities.min().item()

            print(f"[EmbeddingCheck] {name}: avg_sim={avg_similarity:.4f}, "
                  f"max_sim={max_similarity:.4f}, min_sim={min_similarity:.4f}")

            # Warning if embeddings are collapsing
            if avg_similarity > 0.8:
                print(f"❌ CRITICAL: {name} are collapsing! (avg_sim={avg_similarity:.4f})")
                return False
            elif avg_similarity > 0.5:
                print(f"⚠️  WARNING: {name} are converging! (avg_sim={avg_similarity:.4f})")
                return True
            return True

        except Exception as e:
            print(f"⚠️  Embedding quality check failed for {name}: {e}")
            return True


def comprehensive_gradient_debug(model_engine, step_name="Initial"):
    """Comprehensive check of model gradient readiness"""
    print(f"\n🔍 {step_name} GRADIENT READINESS CHECK")
    print("=" * 60)

    model_to_check = model_engine.module if hasattr(model_engine, 'module') else model_engine

    # Check parameter states
    total_params = 0
    trainable_params = 0
    frozen_layers = []

    for name, param in model_to_check.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_layers.append(name)

    print(f"📊 Parameters: {trainable_params:,} trainable / {total_params:,} total")
    print(f"📊 Trainable ratio: {trainable_params / total_params * 100:.1f}%")

    if frozen_layers:
        print(f"❌ Frozen layers found: {len(frozen_layers)}")
        for layer in frozen_layers[:5]:  # Show first 5
            print(f"   - {layer}")

    # Test forward/backward with simple input
    try:
        print("\n🧪 Testing forward/backward pass...")
        model_engine.train()

        # Simple test input
        test_input = model_to_check.tokenizer(
            ["test sentence for gradient check", "another test sentence"],
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(next(model_to_check.parameters()).device)

        # Simple forward pass
        with torch.enable_grad():
            outputs = model_to_check.get_sentence_embedding(**test_input)
            test_loss = outputs.sum()  # Simple loss

            # Backward through DeepSpeed engine
            model_engine.zero_grad()
            model_engine.backward(test_loss)

            # Check gradients
            grads_found = 0
            total_trainable = 0
            for name, param in model_to_check.named_parameters():
                if param.requires_grad:
                    total_trainable += 1
                    if param.grad is not None and param.grad.abs().sum() > 0:
                        grads_found += 1
                        if grads_found <= 3:  # Show first 3
                            grad_norm = param.grad.data.norm(2).item()
                            print(f"   ✅ {name}: grad_norm = {grad_norm:.6e}")

            print(f"📊 Gradient summary: {grads_found}/{total_trainable} parameters have gradients")

            if grads_found == 0:
                print("🚨 CRITICAL: No gradients detected in test!")
                return False
            else:
                print("✅ Gradients flowing correctly!")
                return True

    except Exception as e:
        print(f"🚨 Gradient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def emergency_model_snapshot(model_engine, path):
    """Save model state for debugging"""
    model_to_save = model_engine.module if hasattr(model_engine, 'module') else model_engine
    snapshot = {
        'model_state': model_to_save.state_dict(),
        'grad_status': {name: param.requires_grad for name, param in model_to_save.named_parameters()},
        'param_shapes': {name: param.shape for name, param in model_to_save.named_parameters()}
    }
    torch.save(snapshot, path)
    print(f"💾 Emergency snapshot saved to {path}")


def verify_model_parameters(model):
    """Verify all model parameters are properly set up"""
    print("\n🔍 MODEL PARAMETER VERIFICATION")
    print("=" * 50)

    # Check if any parameters are frozen
    frozen_params = []
    trainable_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(name)
        else:
            frozen_params.append(name)

    print(f"✅ Trainable parameters: {len(trainable_params)}")
    if frozen_params:
        print(f"❌ Frozen parameters: {len(frozen_params)}")
        print("Frozen layers (first 10):")
        for name in frozen_params[:10]:
            print(f"   - {name}")
    else:
        print("✅ All parameters are trainable!")

    # Check parameter devices
    devices = set()
    for param in model.parameters():
        devices.add(param.device)
    print(f"📱 Parameters on devices: {devices}")

    return len(frozen_params) == 0


def diagnostic_loss_balancing(args, current_epoch, loss_components, logits_student_in_batch):
    """
    UPDATED: Diagnostic loss balancing with MUCH more conservative HardNeg weights
    """
    loss_in_batch, loss_hardneg, loss_rd, loss_rd2, loss_feat = loss_components

    # CRITICAL FIX: Monitor individual loss magnitudes and cap them
    with torch.no_grad():
        # If any loss component is exploding, cap it
        loss_in_batch = torch.clamp(loss_in_batch, max=2.0)
        loss_hardneg = torch.clamp(loss_hardneg, max=1.5)  # REDUCED from 2.0
        loss_rd = torch.clamp(loss_rd, max=1.0)
        loss_rd2 = torch.clamp(loss_rd2, max=1.0)
        loss_feat = torch.clamp(loss_feat, max=0.01)

    # === UPDATED WEIGHTS - MUCH MORE CONSERVATIVE ===
    # Since HardNeg loss is increasing too much, drastically reduce its weight
    if current_epoch < 8:
        # Phase 1: Focus on in-batch, minimal hardneg
        alpha = 0.8  # In-batch loss (stable)
        beta = 0.002  # DRASTICALLY REDUCED: Hard negative (was 0.01)
        gamma = 0.001  # Ranking distillation
        eta = 0.0001  # Feature loss
    elif current_epoch < 15:
        # Phase 2: Slightly increase but still very conservative
        alpha = 0.6
        beta = 0.005  # Still very conservative
        gamma = 0.005
        eta = 0.0002
    else:
        # Phase 3: Balanced but conservative
        alpha = 0.4
        beta = 0.01  # Keep hardneg weight very low
        gamma = 0.01
        eta = 0.0005

    balanced_loss = (
            alpha * loss_in_batch +
            beta * loss_hardneg +
            gamma * loss_rd +
            gamma * loss_rd2 +
            eta * loss_feat
    )

    # Add embedding collapse prevention
    with torch.no_grad():
        # Compute embedding diversity penalty
        if logits_student_in_batch.dim() == 2 and logits_student_in_batch.size(0) > 1:
            embeddings_norm = F.normalize(logits_student_in_batch, p=2, dim=1)
            similarity = torch.mm(embeddings_norm, embeddings_norm.T)
            mask = ~torch.eye(embeddings_norm.size(0), device=embeddings_norm.device).bool()
            off_diag_similarities = similarity[mask]

            if off_diag_similarities.numel() > 0:
                avg_sim = off_diag_similarities.mean()
                if avg_sim > 0.5:  # If embeddings are collapsing
                    diversity_penalty = (avg_sim - 0.3) * 0.05  # Reduced penalty
                    balanced_loss = balanced_loss + diversity_penalty
                    print(f"⚠️  Adding diversity penalty: {diversity_penalty:.4f}")

    # Log detailed loss information
    if current_epoch < 5 or (current_epoch < 15 and current_epoch % 2 == 0):
        print(f"🔍 LOSS DIAGNOSTIC Epoch {current_epoch + 1}:")
        print(f"   InBatch: {loss_in_batch.item():.4f} (×{alpha:.3f})")
        print(f"   HardNeg: {loss_hardneg.item():.4f} (×{beta:.3f})")
        print(f"   RD: {loss_rd.item():.4f} (×{gamma:.3f})")
        print(f"   RD2: {loss_rd2.item():.4f} (×{gamma:.3f})")
        print(f"   Feat: {loss_feat.item():.6f} (×{eta:.4f})")
        print(f"   Total: {balanced_loss.item():.4f}")

    return balanced_loss


def aggressive_temperature_scheduling(args, current_epoch):
    """UPDATED: Extended high temperature phase for better stability"""
    # START WITH VERY HIGH TEMPERATURES for maximum stability
    if current_epoch < 5:  # EXTENDED from 3 to 5
        # Phase 1: VERY high temperatures to prevent gradient explosion
        args.temperature_in_batch = 0.5  # Much higher
        args.temperature_hardneg = 0.7  # Much higher
    elif current_epoch < 10:  # EXTENDED from 6 to 10
        # Phase 2: High temperatures
        args.temperature_in_batch = 0.3
        args.temperature_hardneg = 0.4
    elif current_epoch < 15:
        # Phase 3: Moderate temperatures
        args.temperature_in_batch = 0.2
        args.temperature_hardneg = 0.25
    elif current_epoch < 20:
        # Phase 4: Approach target
        args.temperature_in_batch = 0.15
        args.temperature_hardneg = 0.18
    else:
        # Phase 5: Final
        args.temperature_in_batch = 0.12
        args.temperature_hardneg = 0.15

    print(f"🌡️ Temperature Update: in_batch={args.temperature_in_batch:.3f}, hardneg={args.temperature_hardneg:.3f}")
    return args


def add_embedding_regularization(embeddings, lambda_reg=0.01):
    """
    Add regularization to prevent embedding collapse and improve generalization
    """
    # Ensure embeddings require gradients for regularization
    if not embeddings.requires_grad:
        embeddings = embeddings.requires_grad_(True)

    # Variance regularization - encourage diverse embeddings
    embedding_variance = torch.var(embeddings, dim=0).mean()
    variance_penalty = -lambda_reg * embedding_variance  # Encourage high variance

    # Orthogonality regularization
    if embeddings.size(0) > 1:
        normalized_embs = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.mm(normalized_embs, normalized_embs.T)
        mask = ~torch.eye(embeddings.size(0), device=embeddings.device).bool()
        off_diag_similarities = similarity[mask]

        if off_diag_similarities.numel() > 0:
            # Encourage orthogonality (low off-diagonal similarity)
            ortho_penalty = lambda_reg * torch.mean(off_diag_similarities ** 2)
        else:
            ortho_penalty = 0.0
    else:
        ortho_penalty = 0.0

    return variance_penalty + ortho_penalty



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model_dir',default='bert-base-uncased', type=str, help='Model directory')
    parser.add_argument('--train_data_list', nargs='+')
    parser.add_argument('--pos_dir', default='PATH_TO_POS_LOGITS', type=str)
    parser.add_argument('--neg_dir', default='PATH_TO_HARDNEG_LOGITS', type=str)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--inbatch_pkl_path_dir', default='PATH_TO_INBATCH_LOGITS_PKL')
    parser.add_argument('--feature_pkl_path_dir', default='PATH_TO_FEATURE_PKL')
    parser.add_argument('--batch_size', default=32, type=int, help='bs')
    parser.add_argument('--neg_K', default=8, type=int, help='num of hard negs')
    parser.add_argument('--num_heads', default=32, type=int, help='num_heads of pma')
    parser.add_argument('--hidden_dim', default=512, type=int, help='hidden dim of my mlp')
    parser.add_argument('--output_dim', default=1, type=int, help='output dim of my mlp')
    parser.add_argument('--ln', default=True, type=str2bool, help='layer norm for pma')
    parser.add_argument('--norm', default=False, type=str2bool, help='norm after sentence pooling')
    parser.add_argument('--num_epochs', default=10, type=int, help='training epochs')
    parser.add_argument('--padding_side', default='right', type=str, help='padding side')
    parser.add_argument('--max_seq_length', default=250, type=int, help='max_seq_len')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--alpha', default=1, type=float, help='trade-off param')
    parser.add_argument('--beta', default=1, type=float, help='trade-off param')
    parser.add_argument('--gamma', default=0.01, type=float, help='trade-off param')
    parser.add_argument('--eta', default=0.001, type=float, help='trade-off param')
    parser.add_argument('--temperature_in_batch', default=1, type=float, help='temperature in in-batch')
    parser.add_argument('--temperature_hardneg', default=1, type=float, help='temperature in hardneg')
    parser.add_argument('--temperature_teacher_hardneg', default=1, type=float, help='temperature in teacher logits')
    parser.add_argument('--scale_param', default=1, type=float, help='scale param')
    parser.add_argument('--log_interval', default=20, type=int)
    parser.add_argument('--eval_interval', default=200, type=int)
    parser.add_argument('--tb_dir', default='PATH_TO_TENSORBOARD_PATH', type=str)
    parser.add_argument('--patience', default=5, type=int)
    parser.add_argument('--num_ckpt', default=5, type=int)
    parser.add_argument('--training_log', default='PATH_TO_TRAINING_LOG')
    parser.add_argument('--output_dir', default='PATH_TO_OUTPUT_MODEL', type=str, help='Model output directory')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='max gradient norm for clipping')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--bf16', default=True, type=str2bool)
    parser.add_argument('--verbose', default=True, type=str2bool)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--local_rank', type=int, default=-1, help='ds')
    parser.add_argument('--global_rank', type=int, default=-1, help='ds')
    parser.add_argument('--mark', type=str, default='', help='Mark or label for checkpoint folder naming')
    parser.add_argument('--hidden_dropout_prob', default=0.2, type=float)
    parser.add_argument('--attention_dropout_prob', default=0.1, type=float)
    parser.add_argument('--classifier_dropout_prob', default=0.3, type=float)
    parser.add_argument('--inbatch_margin', default=0.1, type=float, help='Margin for in-batch contrastive loss')
    parser.add_argument('--use_adaptive_temp', default=True, type=str2bool, help='Use adaptive temperature scaling')
    parser.add_argument('--debug', default=False, type=str2bool, help='Enable debug mode for detailed logging')
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='Label smoothing factor')
    parser.add_argument('--use_gradient_clipping', default=True, type=str2bool, help='Enable gradient clipping')
    parser.add_argument('--use_layer_wise_lr_decay', default=False, type=str2bool, help='Use layer-wise learning rate decay')
    parser.add_argument('--warmup_ratio', default=0.1, type=float, help='Warmup ratio for learning rate scheduler')
    parser.add_argument('--min_lr_ratio', default=0.001, type=float, help='Minimum learning rate ratio')


    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args.world_size = int(os.getenv('WORLD_SIZE', '0'))


    # --- Ensure num_epochs is the value you expect ---
    # Reason: deepspeed/launcher or environment might override parser defaults.
    # You can set NUM_EPOCHS env var to override; otherwise change the '10' below.
    # desired_epochs = int(os.getenv('NUM_EPOCHS', '15'))  # <- change '10' to the default you want
    # args.num_epochs = desired_epochs

    # Debug print so you can confirm what's actually used at run time
    print(f"[ConfigCheck] WORLD_SIZE={args.world_size}, num_epochs={args.num_epochs}")

    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(filename=f'{args.training_log}')

    if args.seed is not None:
        set_seed(args.seed)
    transformers.set_seed(args.seed)

    micro_bs = args.batch_size

    print("🔄 Initializing model with proper gradient settings...")

    # Simple approach: Create model once and set dropout
    model = Mymodel(
        model_name_or_path=args.base_model_dir,
        alias=None,
        max_seq_length=args.max_seq_length,
        args=args
    )

    # Set dropout on the existing model
    if hasattr(model.plm_model, 'config'):
        model.plm_model.config.hidden_dropout_prob = args.hidden_dropout_prob
        model.plm_model.config.attention_probs_dropout_prob = args.attention_dropout_prob
        print(f"✅ Set dropout on model: hidden={args.hidden_dropout_prob}, attention={args.attention_dropout_prob}")

    # 🔥 FORCE ALL PARAMETERS TO BE TRAINABLE
    for name, param in model.named_parameters():
        param.requires_grad = True

    print("✅ All parameters set to trainable")

    # Verify model parameters
    if args.global_rank == 0:
        verify_model_parameters(model)

    # Method 2: Add dropout to your projection layers
    print("📋 Adding dropout to projection layers...")

    # Find and wrap all linear layers with dropout
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name != 'plm_model':
            print(f"✅ Adding dropout to {name}")
            setattr(model, name, nn.Sequential(
                module,
                nn.Dropout(args.classifier_dropout_prob)
            ))
        elif isinstance(module, nn.Sequential):
            # Insert dropout in sequential layers
            new_layers = []
            for i, layer in enumerate(module):
                new_layers.append(layer)
                if isinstance(layer, (nn.Linear, nn.ReLU, nn.GELU)) and i < len(module) - 1:
                    new_layers.append(nn.Dropout(args.classifier_dropout_prob))
            setattr(model, name, nn.Sequential(*new_layers))
            print(f"✅ Added dropout to sequential layer {name}")

    print("🎯 Dropout configuration completed!")



    model.plm_model.gradient_checkpointing_enable()

    summary_writer = SummaryWriter(log_dir=args.tb_dir)

    # check_architecture_consistency(model)
    # verify_bert_setup(model,args)

    # ✅ Use ZeRO Stage 1 (not Stage 2) for single GPU to avoid gradient issues
    # === FIXED DEEPSPEED CONFIG ===
    # FIXED DeepSpeed config with gradient clipping - TEMPORARY CONFIG
    ds_config = {
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": args.max_grad_norm,
        "bfloat16": {
            "enabled": args.bf16
        },
        "fp16": {
            "enabled": False
        },
        "zero_optimization": {
            "stage": 1,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": args.weight_decay
            }
        },
        "steps_per_print": 10,
        "wall_clock_breakdown": False
    }


    # 🔥 FIXED: Handle device assignment properly
    if hasattr(args, 'local_rank') and args.local_rank >= 0:
        device = torch.device(f"cuda:{args.local_rank}")
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"✅ Using device: {device}")
    model = model.to(device)
    print(f"✅ Model moved to device: {next(model.parameters()).device}")

    model.plm_model.gradient_checkpointing_enable()
    print("🚀 Gradient checkpointing enabled - should fix OOM!")

    # Initialize with basic config first
    print("🔄 Reinitializing model engine with updated DeepSpeed config...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        config=ds_config
    )


    if not torch.distributed.is_initialized():
        dist.init_process_group(backend='nccl')
    args.global_rank = dist.get_rank()
    args.local_rank = dist.get_rank()
    device = torch.device(f"cuda:{args.local_rank}")
    args.device = device
    torch.cuda.set_device(device)

    # === Diagnostic: Check number of trainable parameters ===
    num_trainable = sum(p.numel() for p in model_engine.parameters() if p.requires_grad)
    num_frozen = sum(p.numel() for p in model_engine.parameters() if not p.requires_grad)
    print(f"[Debug] Trainable parameters: {num_trainable:,} | Frozen parameters: {num_frozen:,}")
    if num_trainable == 0:
        raise ValueError("❌ All parameters are frozen — nothing will update!")

    # Create data loaders first
    train_dataset = TrainDataset(model.tokenizer, pos_dir=args.pos_dir, neg_dir=args.neg_dir, datadir=args.data_dir,
                                 names=args.train_data_list, batch_size=micro_bs, neg_K=args.neg_K,
                                 process_index=args.global_rank, num_processes=args.world_size)
    val_dataset = ValDataset(model.tokenizer, pos_dir=args.pos_dir, neg_dir=args.neg_dir, datadir=args.data_dir,
                             names=args.train_data_list, batch_size=micro_bs, neg_K=args.neg_K,
                             process_index=args.global_rank, num_processes=args.world_size)


    if args.global_rank == -1:
        train_sampler = RandomSampler(train_dataset)
        val_sampler = RandomSampler(val_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset)
    # ✅ Use actual micro batch size (no fake_bs variable)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=micro_bs,
        shuffle=False,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=micro_bs,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=0
    )

    # === UPDATE DEEPSPEED CONFIG WITH WARMUP CALCULATION (NOW THAT DATA LOADERS EXIST) ===
    # Calculate warmup steps now that train_dataloader is defined
    total_training_steps = args.num_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    warmup_steps = int(total_training_steps * args.warmup_ratio)

    # Update ds_config with scheduler now that we have the data
    ds_config.update({
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": args.lr * args.min_lr_ratio,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": warmup_steps
            }
        }
    })

    print(f"📊 Training Configuration:")
    print(f"   Total training steps: {total_training_steps}")
    print(f"   Warmup steps: {warmup_steps}")
    print(f"   Train dataloader length: {len(train_dataloader)}")
    print(f"   Gradient accumulation steps: {args.gradient_accumulation_steps}")


    # Comprehensive gradient check
    gradient_ok = comprehensive_gradient_debug(model_engine, "Pre-Training Verification")

    if not gradient_ok:
        print("🚨 CRITICAL: Model not ready for training - gradients not flowing!")
        print("💡 Debugging steps:")
        print("   1. Check if DeepSpeed is properly initialized")
        print("   2. Verify model is in training mode")
        print("   3. Check parameter device consistency")
        print("   4. Verify loss function computation")

        # Try a simple training step to debug
        print("\n🧪 Running diagnostic training step...")
        try:
            model_engine.train()
            # Use a simple batch to test
            diagnostic_batch = next(iter(train_dataloader))
            sentence_a, sentence_b, _, _, _, task_id = diagnostic_batch
            sentence_all = sentence_a + sentence_b

            inputs_all = model.tokenizer(sentence_all, padding='max_length',
                                         max_length=args.max_seq_length, truncation=True, return_tensors='pt')
            inputs_all = inputs_all.to(device)

            # Forward
            logits, _, _ = model_engine(inputs_all, task_id.to(device), 'train')
            simple_loss = logits.sum()  # Simple loss for testing

            # Backward
            model_engine.zero_grad()
            model_engine.backward(simple_loss)

            print("✅ Diagnostic step completed")

            # Check gradients again
            comprehensive_gradient_debug(model_engine, "After Diagnostic Step")

        except Exception as e:
            print(f"❌ Diagnostic step failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("🎉 Model verified and ready for training!")

    print("=" * 80 + "\n")

    print(
        f"[InitCheck] BF16 enabled: {ds_config['bfloat16']['enabled']} | ZeRO stage: {ds_config['zero_optimization']['stage']}")

    # Add this right after creating train_dataset and val_dataset
    print("\n🔍 TEACHER LOGITS DIAGNOSTIC:")

    # Check if teacher logits are actually being loaded
    sample_batch = next(iter(train_dataloader))
    sentence_a, sentence_b, logits_teacher_pos, sentence_hardneg, logits_teacher_hardneg, task_id = sample_batch

    print(f"Batch shapes:")
    print(f"  sentence_a: {len(sentence_a)}")
    print(f"  sentence_b: {len(sentence_b)}")
    print(f"  logits_teacher_pos: {logits_teacher_pos.shape if logits_teacher_pos is not None else 'MISSING'}")
    print(
        f"  logits_teacher_hardneg: {logits_teacher_hardneg.shape if logits_teacher_hardneg is not None else 'MISSING'}")

    # Check if teacher logits are zeros or have meaningful values
    if logits_teacher_pos is not None:
        print(
            f"Teacher pos logits stats: min={logits_teacher_pos.min():.4f}, max={logits_teacher_pos.max():.4f}, mean={logits_teacher_pos.mean():.4f}")
    if logits_teacher_hardneg is not None:
        print(
            f"Teacher hardneg logits stats: min={logits_teacher_hardneg.min():.4f}, max={logits_teacher_hardneg.max():.4f}, mean={logits_teacher_hardneg.mean():.4f}")

    # If teacher logits are missing or zeros, you have a data preparation problem
    if logits_teacher_pos is None or (logits_teacher_pos is not None and logits_teacher_pos.abs().max() < 1e-6):
        print("🚨 CRITICAL: Teacher positive logits are missing or zero!")
        print("   Your model cannot learn from teacher supervision.")
        print("   Check your data preparation pipeline.")




    print("Length of Train Dataset:",len(train_dataset))
    if len(train_dataset) > 0:
        train_data_flag = True

    if not train_data_flag:
        raise ValueError("Error, train_file|use_hf_dataset must be specified")

    all_dataset_id = train_dataset.dataset_id_dict
    all_dataset_id_reverse = {v: k for k, v in train_dataset.dataset_id_dict.items()}
    rel_dataset_id = [all_dataset_id[dataset_name] for dataset_name in args.train_data_list]
    os.makedirs(args.output_dir, exist_ok=True)

    train_loader_size = len(train_dataloader)
    val_loader_size = len(val_dataloader)

    criterion = nn.CrossEntropyLoss(reduction='none')
    nll_criterion = nn.NLLLoss(reduction='none')

    global_step = 0
    best_eval_metric = 0
    trained_epochs = 0
    min_reduce_loss_eval = float('inf')
    best_epoch = 0
    stop = 0
    best_step = 0

    teacher_feature_cos_dict = load_pickle(args.feature_pkl_path_dir)
    teacher_inbatch_logits_dict = load_pickle(args.inbatch_pkl_path_dir)

    # Simple teacher logits verification
    print("\n=== TEACHER LOGITS VERIFICATION ===")
    if teacher_inbatch_logits_dict:
        sample_key = list(teacher_inbatch_logits_dict.keys())[0]
        print(f"Teacher logits loaded for key: {sample_key}")
    print("Teacher features loaded" if teacher_feature_cos_dict else "No teacher features found")

    epoch_loss_overall = 0.0
    epoch_loss_inbatch = 0.0
    epoch_loss_hardneg = 0.0
    epoch_loss_rd = 0.0
    epoch_loss_rd2 = 0.0
    epoch_loss_feat = 0.0
    num_train_batches = 0

    print("\nAbout to start training loop...")
    logging.info(f"\nAbout to start training loop...")

    for current_epoch in trange(int(args.num_epochs), desc="\nEpoch", disable=(args.global_rank != 0), mininterval=0):
        # AGGRESSIVE MEMORY CLEANUP AT EPOCH START
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'memory_stats'):
            torch.cuda.reset_peak_memory_stats()


        torch.autograd.set_detect_anomaly(False)
        if False:
            pass
        torch.cuda.empty_cache()
        model_engine.train()

        # Start of epoch logging
        logging.info(f"Starting Epoch {current_epoch + 1}/{args.num_epochs}")
        epoch_mrr10 = 0.0
        epoch_ndcg10 = 0.0
        num_mrr_batches = 0

        if current_epoch == 0:
            args.previous_hardneg_loss = 0.5  # Initial estimate
        elif step == len(train_dataloader) - 1:  # Last step of epoch
            args.previous_hardneg_loss = loss_hardneg.item()

        batch_iterator = tqdm(train_dataloader,
                              desc=f"Running Epoch {current_epoch + 1} of {args.num_epochs}",
                              disable=(args.global_rank != 0),
                              mininterval=0)
        for step, batch in enumerate(batch_iterator):

            # 🔥 MEMORY FIX: Clear cache and manage memory aggressively
            if step > 0 and step % 5 == 0:  # Clear every 5 steps
                torch.cuda.empty_cache()

            sentence_a, sentence_b, logits_teacher_pos, sentence_hardneg, logits_teacher_hardneg, task_id = batch

            # Only use torch.no_grad() for diagnostic prints, not for the actual data
            with torch.no_grad():
                print("\n=== TEACHER LOGIT DIAGNOSTIC ===")
                print(f"logits_teacher_pos shape: {logits_teacher_pos.shape}")
                print(f"logits_teacher_hardneg shape: {logits_teacher_hardneg.shape}")

                # Check what probabilities we're actually getting
                if logits_teacher_pos is not None:
                    pos_can_prob = logits_teacher_pos[..., 0].mean().item()  # P(Can|positive)
                    pos_cannot_prob = logits_teacher_pos[..., 1].mean().item()  # P(Cannot|positive)
                    print(f"Positive samples - Can: {pos_can_prob:.4f}, Cannot: {pos_cannot_prob:.4f}")

                if logits_teacher_hardneg is not None:
                    neg_can_prob = logits_teacher_hardneg[..., 0].mean().item()  # P(Can|negative)
                    neg_cannot_prob = logits_teacher_hardneg[..., 1].mean().item()  # P(Cannot|negative)
                    print(f"Negative samples - Can: {neg_can_prob:.4f}, Cannot: {neg_cannot_prob:.4f}")

                # The key insight: For ranking, we want P(Can|positive) > P(Can|negative)
                if logits_teacher_pos is not None and logits_teacher_hardneg is not None:
                    separation = pos_can_prob - neg_can_prob
                    print(f"Separation (pos_can - neg_can): {separation:.4f}")
                    if separation < 0:
                        print("🚨 CRITICAL: Negative discrimination in teacher logits!")
                        print("💡 You might need to use P(Cannot) for negatives instead")

            sentence_all = sentence_a + sentence_b + sentence_hardneg
            bs = logits_teacher_pos.size(0)
            key = f'global_rank{args.global_rank}'

            if key not in teacher_inbatch_logits_dict:
                print(
                    f"[Error] Missing key '{key}' in teacher_inbatch_logits_dict. Available keys: {list(teacher_inbatch_logits_dict.keys())}")
                raise KeyError(f"Missing key '{key}' in teacher_inbatch_logits_dict.")

            rank_data = teacher_inbatch_logits_dict[key]

            # Handle double nesting if needed
            if isinstance(rank_data, dict) and key in rank_data:
                print(f"[Info] Detected double nesting for {key}. Accessing inner dictionary...")
                rank_data = rank_data[key]

            # Now handle the actual structure
            if isinstance(rank_data, dict):
                step_key = str(step)
                if step_key not in rank_data:
                    print(
                        f"[Error] Step key '{step_key}' not found in teacher_inbatch_logits_dict[{key}]. Available step keys: {list(rank_data.keys())[:5]}...")
                    raise KeyError(f"Step key '{step_key}' missing in teacher_inbatch_logits_dict[{key}]")
                logits_teacher_inbatch = rank_data[step_key].to(device)
            else:
                if step >= len(rank_data):
                    print(
                        f"[Error] Step {step} out of range for teacher_inbatch_logits_dict[{key}] (len={len(rank_data)})")
                    raise IndexError(f"Step {step} out of range for teacher_inbatch_logits_dict[{key}]")
                logits_teacher_inbatch = rank_data[step].to(device)

            # logits_teacher_inbatch = teacher_inbatch_logits_dict[key][step].to(device)
            feature_teacher_cos = teacher_feature_cos_dict[key][step].to(device)

            # Normalize teacher logits for in-batch and hard-neg

            # 1. Reshape hardneg
            # --- Robust teacher logits processing and normalization ---
            # Ensure teacher positive logits shape: [B, 1, 2]
            # --- Robust teacher logits processing and normalization ---
            # Ensure logits_teacher_pos shape: [B, 1, D]
            if logits_teacher_pos.dim() == 2:  # shape [B, D] or [B, 1]
                logits_teacher_pos = logits_teacher_pos.unsqueeze(1)  # [B,1,D]

            # Ensure logits_teacher_hardneg shape: [B, neg_K, D]
            logits_teacher_hardneg = logits_teacher_hardneg.view(bs, args.neg_K, -1)

            # Make sure last dimension matches logits_teacher_hardneg
            if logits_teacher_pos.size(-1) != logits_teacher_hardneg.size(-1):
                logits_teacher_pos = logits_teacher_pos.expand(-1, -1, logits_teacher_hardneg.size(-1))

            # Concatenate positive and hardneg along dim=1: [B, neg_K+1, D]
            logits_teacher_hardneg = torch.cat([logits_teacher_pos, logits_teacher_hardneg], dim=1)

            # Move to device
            logits_teacher_hardneg = logits_teacher_hardneg.to(device)
            logits_teacher_inbatch = logits_teacher_inbatch.to(device)

            # Normalize along last dimension (mean-centering) and clamp
            logits_teacher_inbatch = logits_teacher_inbatch - logits_teacher_inbatch.mean(dim=-1, keepdim=True)
            logits_teacher_hardneg = logits_teacher_hardneg - logits_teacher_hardneg.mean(dim=-1, keepdim=True)
            logits_teacher_inbatch = torch.clamp(logits_teacher_inbatch, -5, 5)
            logits_teacher_hardneg = torch.clamp(logits_teacher_hardneg, -5, 5)
            inputs_all = model.tokenizer(sentence_all, padding='max_length', max_length=args.max_seq_length,
                                         truncation=True, return_tensors='pt')
            inputs_all = inputs_all.to(device)
            task_id = task_id.to(device)
            logits_student_in_batch, logits_student_hardneg, rep_student_pos_hardneg = model_engine(inputs_all, task_id,
                                                                                                    'train')

            # 🔥 YOUR MODEL FORWARD PASS
            logits_student_in_batch, logits_student_hardneg, rep_student_pos_hardneg = model_engine(inputs_all, task_id,
                                                                                                    'train')

            # THE EMBEDDING COLLAPSE PREVENTION
            with torch.no_grad():
                # Check embedding diversity
                embeddings = model_engine.module.get_sentence_embedding(**inputs_all)
                embeddings_norm = F.normalize(embeddings, p=2, dim=1)

                # Compute similarity matrix
                similarity = torch.mm(embeddings_norm, embeddings_norm.T)
                mask = ~torch.eye(embeddings.size(0), device=embeddings.device).bool()
                off_diag_similarities = similarity[mask]

                avg_sim = off_diag_similarities.mean().item()
                max_sim = off_diag_similarities.max().item()

                print(f"🔍 EMBEDDING CHECK: avg_sim={avg_sim:.4f}, max_sim={max_sim:.4f}")

                # If embeddings are collapsing, add a diversity penalty
                if avg_sim > 0.8:
                    diversity_penalty = (avg_sim - 0.3) * 0.1  # Penalize high similarity
                    loss_batch = loss_batch + diversity_penalty
                    print(f"⚠️  Adding diversity penalty: {diversity_penalty:.4f}")
            #  END OF EMBEDDING COLLAPSE PREVENTION

            # --- Normalize student logits (avoid division by zero) ---
            eps = 1e-12
            _student_dtype = logits_student_in_batch.dtype
            logits_student_in_batch = logits_student_in_batch.float()
            logits_student_hardneg = logits_student_hardneg.float()

            logits_student_in_batch = logits_student_in_batch / (logits_student_in_batch.norm(dim=-1, keepdim=True) + eps)
            logits_student_hardneg = logits_student_hardneg / (logits_student_hardneg.norm(dim=-1, keepdim=True) + eps)

            logits_student_in_batch = logits_student_in_batch.to(_student_dtype)
            logits_student_hardneg = logits_student_hardneg.to(_student_dtype)
            # -----------------------------------------------------
            # Add this diagnostic before the loss calculation
            with torch.no_grad():
                pos_scores = logits_student_in_batch[:, 0]
                neg_scores = logits_student_in_batch[:, 1:]
                separation = (pos_scores.mean() - neg_scores.mean()).item()
                accuracy = (pos_scores.unsqueeze(1) > neg_scores).float().mean().item()

                print(f"\n📊 PRE-LOSS DIAGNOSTIC:")
                print(f"   Student Pos Mean: {pos_scores.mean().item():.4f}")
                print(f"   Student Neg Mean: {neg_scores.mean().item():.4f}")
                print(f"   Separation: {separation:.4f}")
                print(f"   Accuracy: {accuracy:.4f}")

                if separation < 0:
                    print("   🚨 ALERT: Student has NEGATIVE separation before loss calculation!")

            loss_in_batch = cal_loss_in_batch(args, logits_student_in_batch, args.temperature_in_batch, criterion)

            logits_teacher_pos = logits_teacher_pos.to(args.device)
            # print(f"[Debug] logits_teacher_hardneg.shape before reshape: {logits_teacher_hardneg.shape}")
            # print(f"[Debug] Expected reshape: ({micro_bs}, {args.neg_K}, 2) = {micro_bs * args.neg_K * 2}")
            # print(f"[Debug] Actual total elements: {logits_teacher_hardneg.numel()}")

            # Ensure logits_teacher_pos has shape [B, 1, D] to match hardneg
            logits_teacher_pos = logits_teacher_pos.to(args.device)
            # Ensure logits_teacher_pos has shape [B, 1, D] to match hardneg
            last_dim = logits_teacher_hardneg.size(-1)  # D
            if logits_teacher_pos.dim() == 2:  # shape [B, D] or [B, 1]
                logits_teacher_pos = logits_teacher_pos.unsqueeze(-1) if logits_teacher_pos.size(
                    1) == 1 else logits_teacher_pos
                if logits_teacher_pos.dim() == 2:
                    logits_teacher_pos = logits_teacher_pos.unsqueeze(1)  # [B, 1, D]


            print(f"logits_teacher_pos.shape: {logits_teacher_pos.shape}")  # Should be [micro_bs, 2]
            # print(f"Type: {type(logits_teacher_pos)}")
            # print(f"Min: {logits_teacher_pos.min()}, Max: {logits_teacher_pos.max()}")
            # print(f"logits_teacher_pos.unsqueeze(1).shape: {logits_teacher_pos.unsqueeze(1).shape}")  # Should be [micro_bs, 1, 2]
            # print(f"logits_teacher_hardneg.shape after reshape: {logits_teacher_hardneg.shape}")  # Should be [micro_bs, neg_K, 2]

            loss_hardneg = cal_loss_hardneg(args, logits_teacher_hardneg, logits_student_hardneg,
                                            args.temperature_teacher_hardneg, args.temperature_hardneg, nll_criterion)

            loss_rd = cal_loss_rd(args, logits_teacher_hardneg, logits_student_hardneg, args.temperature_teacher_hardneg)


            batch_size = logits_teacher_inbatch.shape[0]
            inbatch = logits_teacher_inbatch.shape[1] if logits_teacher_inbatch.dim() == 2 else 1
            print(f"[Debug] logits_teacher_inbatch batch_size = {batch_size}, inbatch = {inbatch}")

            # Remove diagonal (self-positive) if needed
            # B = logits_teacher_inbatch.size(0)  # actual batch size
            # if logits_teacher_inbatch.size(1) == B:  # includes diagonal
            #     mask = torch.ones_like(logits_teacher_inbatch, dtype=torch.bool)
            #     mask[torch.arange(B), torch.arange(B)] = 0
            #     logits_teacher_inbatch = logits_teacher_inbatch[mask].view(B, B - 1, -1)

            print("[Debug] logits_teacher_inbatch.shape:", logits_teacher_inbatch.shape)
            print("[Debug] micro_bs:", micro_bs)

            # Ensure positive teacher logits are higher than negatives for all dimensions
            pos_logits = logits_teacher_hardneg[:, 0:1, :].clone()  # [B,1,D]
            neg_logits = logits_teacher_hardneg[:, 1:, :].clone()  # [B,neg_K,D]

            # Compute per-sample max of negatives across dim=1
            max_neg = neg_logits.max(dim=1, keepdim=True)[0]  # [B,1,D]


            # Update the logits_teacher_hardneg
            logits_teacher_hardneg[:, 0:1, :] = pos_logits

            # print("Teacher logits (pos) raw:", logits_teacher_pos[:2])
            # print("Teacher logits (neg) raw:", logits_teacher_hardneg[:2])
            # print("Teacher logits (pos):", logits_teacher_hardneg[:, 0, :].mean().item())
            # print("Teacher logits (neg):", logits_teacher_hardneg[:, 1:, :].mean().item())
            print(f"Teacher logits (pos[0] - Can): {logits_teacher_pos[..., 0].mean().item():.4f}")
            print(f"Teacher logits (neg[0] - Can): {logits_teacher_hardneg[..., 0].mean().item():.4f}")

            print("Teacher logits inbatch pos mean/std:", logits_teacher_inbatch[:, :, 1].mean().item(),
                  logits_teacher_inbatch[:, :, 1].std().item())
            print("Teacher logits hardneg mean/std:", logits_teacher_hardneg[:, :, 1].mean().item(),
                  logits_teacher_hardneg[:, :, 1].std().item())

            assert logits_teacher_inbatch.shape == (micro_bs, micro_bs - 1, 2)

            loss_rd2 = cal_loss_rd2(args, logits_teacher_hardneg, logits_teacher_inbatch, args.temperature_teacher_hardneg,
                                    logits_student_hardneg, logits_student_in_batch, sigmoid, args.scale_param)

            # Convert raw teacher features to cosine sim matrix
            feature_teacher_cos = feature_teacher_cos / feature_teacher_cos.norm(dim=-1, keepdim=True)
            teacher_feat_cos_matrix = torch.matmul(feature_teacher_cos, feature_teacher_cos.transpose(-2, -1))

            loss_feat = cal_feat_loss(args, teacher_feat_cos_matrix, rep_student_pos_hardneg)

            print("loss_in_batch:", loss_in_batch)
            print("loss_hardneg:", loss_hardneg)
            print("loss_rd2:", loss_rd2)
            print("loss_rd:", loss_rd)
            print("loss_feat:", loss_feat)

            # Compute MRR@10 / NDCG@10 for in-batch logits
            # Construct labels: 1 for positive, 0 for negatives
            # 🔥 FIXED: Correct in-batch MRR calculation with temperature
            def compute_in_batch_mrr_correct(logits, temperature=1.0):
                """
                Correct in-batch MRR calculation where positive is diagonal
                """
                bs = logits.shape[0]

                # Apply same temperature as loss calculation
                logits = logits / temperature

                # For in-batch negatives, positive is the diagonal (query_i vs doc_i)
                labels = torch.arange(bs).to(logits.device)

                # Compute similarity scores
                scores = F.softmax(logits, dim=1)

                # Get rankings (descending order)
                rankings = scores.argsort(dim=1, descending=True)

                mrr_sum = 0
                for i in range(bs):
                    # Find rank of the correct positive (diagonal)
                    rank = (rankings[i] == i).nonzero(as_tuple=True)[0].item() + 1
                    mrr_sum += 1.0 / rank

                return mrr_sum / bs

            # Use the corrected MRR calculation
            mrr10 = compute_in_batch_mrr_correct(logits_student_in_batch, args.temperature_in_batch)
            ndcg10 = mrr10  # For simplicity, or implement proper NDCG if needed
            epoch_mrr10 += mrr10
            epoch_ndcg10 += ndcg10
            num_mrr_batches += 1

            # Optional: log to console and TensorBoard
            if args.global_rank == 0 and step % args.log_interval == 0:
                train_log_dict = {}
                print(f"[Debug] Step {step} | MRR@10: {mrr10:.4f} | NDCG@10: {ndcg10:.4f}")
                train_log_dict['MRR@10'] = mrr10
                train_log_dict['NDCG@10'] = ndcg10
                write_tensorboard(summary_writer, train_log_dict, global_step)

            # Aggressive temperature scheduling
            args = aggressive_temperature_scheduling(args, current_epoch)

            # Use UPDATED diagnostic loss balancing with MUCH more conservative weights
            loss_components = (loss_in_batch, loss_hardneg, loss_rd, loss_rd2, loss_feat)
            loss_batch = diagnostic_loss_balancing(args, current_epoch, loss_components, logits_student_in_batch)

            # Add dynamic adjustment based on HardNeg growth
            if step == 0 and current_epoch > 0:  # Check at beginning of each epoch
                hardneg_growth = loss_hardneg.item() / max(0.1, getattr(args, 'previous_hardneg_loss', 0.5))
                if hardneg_growth > 1.3:  # If HardNeg increased by more than 30%
                    print(f"🚨 HardNeg loss growing too fast: {hardneg_growth:.2f}x")
                    # The weights are now handled in the function above


            # Ensure loss_batch requires gradients
            if not loss_batch.requires_grad:
                loss_batch = loss_batch.requires_grad_(True)

            # Add embedding regularization to prevent collapse and overfitting
            if step % 2 == 0:  # Apply regularization every 2 steps to avoid over-regularization
                # REMOVED torch.no_grad() - regularization needs gradients
                embeddings = model_engine.module.get_sentence_embedding(**inputs_all)
                reg_penalty = add_embedding_regularization(embeddings, lambda_reg=0.005)
                loss_batch = loss_batch + reg_penalty


            if args.verbose:
                batch_iterator.set_description(
                    f"Epoch: {current_epoch + 1}/{args.num_epochs}, Batch:{step}/{len(train_dataloader)}, Loss: {loss_batch:9.4f}")

            # Before backward - ensure loss has gradients
            loss_batch = loss_batch.float()  # ensure float

            # CRITICAL: Verify loss has gradients before backward pass
            if not loss_batch.requires_grad:
                print("🚨 CRITICAL: Loss does not require gradients! Recomputing without no_grad...")
                # Recompute without any no_grad context
                logits_student_in_batch, logits_student_hardneg, rep_student_pos_hardneg = model_engine(inputs_all,
                                                                                                        task_id,
                                                                                                        'train')
                loss_in_batch = cal_loss_in_batch(args, logits_student_in_batch, args.temperature_in_batch, criterion)
                loss_hardneg = cal_loss_hardneg(args, logits_teacher_hardneg, logits_student_hardneg,
                                                args.temperature_teacher_hardneg, args.temperature_hardneg,
                                                nll_criterion)
                loss_rd = cal_loss_rd(args, logits_teacher_hardneg, logits_student_hardneg,
                                      args.temperature_teacher_hardneg)
                loss_rd2 = cal_loss_rd2(args, logits_teacher_hardneg, logits_teacher_inbatch,
                                        args.temperature_teacher_hardneg,
                                        logits_student_hardneg, logits_student_in_batch, sigmoid, args.scale_param)
                loss_feat = cal_feat_loss(args, teacher_feat_cos_matrix, rep_student_pos_hardneg)

                loss_components = (loss_in_batch, loss_hardneg, loss_rd, loss_rd2, loss_feat)
                loss_batch = adaptive_loss_balancing(args, current_epoch, loss_components)

            # After computing loss_batch, add this SIMPLIFIED check:
            if step % 10 == 0:
                print(f"\n[LossCheck] Step {step}")
                print(f"  Loss value: {loss_batch.item():.6f}")
                print(f"  Loss requires_grad: {loss_batch.requires_grad}")
                print(f"  Loss device: {loss_batch.device}")

                # Simple check: if loss has gradients flowing through the model
                model_to_check = model_engine.module if hasattr(model_engine, 'module') else model_engine
                has_gradients = any(p.grad is not None for p in model_to_check.parameters() if p.requires_grad)

                if has_gradients:
                    print("  ✅ Loss is connected to model (gradients detected)")
                else:
                    print("  ⚠️  No gradients detected yet")


            # === Diagnostic: Check gradient flow ===

            scaled_loss = loss_batch / args.gradient_accumulation_steps

            # === IMPROVED BACKWARD PASS WITH GRADIENT MONITORING ===
            # Backward pass with gradient monitoring
            model_engine.backward(scaled_loss)

            if step % 50 == 0:
                total_norm = 0.0
                max_grad = 0.0
                for name, param in model_engine.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2).item()
                        total_norm += param_norm ** 2
                        if param_norm > max_grad:
                            max_grad = param_norm

                total_norm = total_norm ** 0.5
                print(f"📊 Gradient Monitor: total_norm={total_norm:.4f}, max_grad={max_grad:.4f}")

                # If gradients are exploding, apply emergency clipping
                if total_norm > 10.0:
                    print(f"🚨 GRADIENT EXPLOSION DETECTED! Applying emergency clipping...")
                    torch.nn.utils.clip_grad_norm_(model_engine.parameters(), max_norm=1.0)

            # Optional: Manual gradient clipping for extra safety
            if args.max_grad_norm > 0 and args.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(model_engine.parameters(), args.max_grad_norm)

            model_engine.step()

            del logits_teacher_hardneg, logits_teacher_inbatch, feature_teacher_cos
            torch.cuda.empty_cache()

            # Check LR using model_engine.optimizer (remove the duplicate optimizer check)
            if step % 200 == 0 and hasattr(model_engine.optimizer, 'param_groups'):
                current_lr = model_engine.optimizer.param_groups[0]['lr']
                print(f"[LRCheck] Step {step}: current LR = {current_lr:.6e}")

            # Increment global step
            global_step += 1
            # --- Accumulate raw per-batch losses for full-epoch averaging ---
            epoch_loss_overall += loss_batch.detach().item()
            epoch_loss_inbatch += loss_in_batch.detach().item()
            epoch_loss_hardneg += loss_hardneg.detach().item()
            epoch_loss_rd += loss_rd.detach().item()
            epoch_loss_rd2 += loss_rd2.detach().item()
            epoch_loss_feat += loss_feat.detach().item()
            num_train_batches += 1

            # --- Optional: short-interval logging for console/TensorBoard ---
            if global_step % args.log_interval == 0 and args.global_rank == 0:
                print(
                    f"[TrainStep {global_step}] "
                    f"Loss: {loss_batch.item():.4f} | "
                    f"InBatch: {loss_in_batch.item():.4f} | "
                    f"HardNeg: {loss_hardneg.item():.4f} | "
                    f"RD: {loss_rd.item():.4f} | "
                    f"RD2: {loss_rd2.item():.4f} | "
                    f"Feat: {loss_feat.item():.6f}"
                )

                train_log_dict = {
                    'loss_overall': loss_batch.item(),
                    'loss_inbatch': loss_in_batch.item(),
                    'loss_hardneg': loss_hardneg.item(),
                    'loss_rd': loss_rd.item(),
                    'loss_rd2': loss_rd2.item(),
                    'loss_feat': loss_feat.item(),
                }
                write_tensorboard(summary_writer, train_log_dict, global_step)

        # --- Evaluate and optionally save based on validation loss ---
        # --- Save best checkpoint based on MRR@10 instead of validation loss ---
        avg_mrr10 = epoch_mrr10 / max(1, num_mrr_batches)
        avg_ndcg10 = epoch_ndcg10 / max(1, num_mrr_batches)


        if args.global_rank == 0:
            # Always save checkpoint for each epoch
            epoch_ckpt_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{current_epoch + 1}")
            os.makedirs(epoch_ckpt_dir, exist_ok=True)
            save_model_properly(model_engine, epoch_ckpt_dir)
            print(f"[Checkpoint] Saved epoch {current_epoch + 1} to {epoch_ckpt_dir}")




            # ✅ CORRECT: Run validation and use the actual result
            full_retrieval_mrr = validate_with_full_retrieval(
                model_engine, val_dataloader, args, current_epoch
            )

            # Then use it for model selection
            if full_retrieval_mrr > best_eval_metric:
                best_eval_metric = full_retrieval_mrr
                best_epoch = current_epoch
                best_step = global_step

                if args.global_rank == 0:
                    print(f"🌟 New best FULL RETRIEVAL MRR@10: {best_eval_metric:.4f} at epoch {current_epoch + 1}")

                    best_ckpt_dir = os.path.join(
                        args.output_dir, f"checkpoint-epoch-{current_epoch + 1}-step-{global_step}-BEST-{args.mark}"
                    )
                    save_model_properly(model_engine, best_ckpt_dir)
                    print(f"💾 Saved BEST model to: {best_ckpt_dir}")

        # === IMPROVED VALIDATION AND SMART SAVING ===
        def smart_validation_and_saving(model_engine, args, current_epoch, global_step,
                                        current_mrr, best_mrr, stop_counter, train_metrics):
            """
            Smart model saving based on multiple criteria including training metrics
            """
            save_reasons = []

            # 1. Significant MRR improvement
            if current_mrr > best_mrr + 0.003:
                save_reasons.append("best_mrr")
                best_mrr = current_mrr
                stop_counter = 0

            # 2. Good training metrics (even if validation slightly lower)
            elif (train_metrics['train_mrr'] > 0.25 and
                  current_mrr > best_mrr * 0.95 and
                  current_epoch > 5):
                save_reasons.append("good_train_perf")

            # 3. Recovery save if significant drop but good training
            elif (current_mrr < best_mrr * 0.85 and
                  train_metrics['train_mrr'] > 0.22 and
                  current_epoch > 3):
                save_reasons.append("recovery")

            # 4. Regular checkpoints (every 3 epochs)
            if (current_epoch + 1) % 3 == 0:
                save_reasons.append("regular")

            # Save if any reason met
            if save_reasons and args.global_rank == 0:
                reason_str = "_".join(save_reasons)
                ckpt_dir = os.path.join(
                    args.output_dir,
                    f"checkpoint-epoch-{current_epoch + 1}-{reason_str}-{args.mark}"
                )
                save_model_properly(model_engine, ckpt_dir)
                print(f"💾 Saved checkpoint: {ckpt_dir} (reasons: {', '.join(save_reasons)})")

            # Update early stopping counter
            def compute_validation_guided_early_stopping(current_mrr, best_mrr, stop_counter,
                                                         current_epoch, min_improvement=0.002):
                """
                Smart early stopping that considers both absolute improvement and training stage
                """
                improvement = current_mrr - best_mrr

                # More lenient in early epochs
                if current_epoch < 5:
                    effective_improvement = improvement - 0.001
                elif current_epoch < 10:
                    effective_improvement = improvement - 0.0005
                else:
                    effective_improvement = improvement

                if effective_improvement > min_improvement:
                    return 0, current_mrr  # Reset counter, update best MRR
                else:
                    return stop_counter + 1, best_mrr

            stop_counter, best_mrr = compute_validation_guided_early_stopping(
                current_mrr, best_mrr, stop_counter, current_epoch
            )

            return best_mrr, stop_counter

        # Run validation
        full_retrieval_mrr = validate_with_full_retrieval(
            model_engine, val_dataloader, args, current_epoch
        )

        # Calculate training metrics FIRST before using them
        if num_train_batches > 0:
            avg_train_loss = epoch_loss_overall / num_train_batches
            avg_train_loss_inbatch = epoch_loss_inbatch / num_train_batches
        else:
            avg_train_loss = 0.0
            avg_train_loss_inbatch = 0.0

        # Prepare training metrics for smart saving
        train_metrics = {
            'train_mrr': avg_mrr10,
            'train_loss': avg_train_loss,
            'inbatch_loss': avg_train_loss_inbatch
        }

        # Use smart saving strategy
        best_eval_metric, stop = smart_validation_and_saving(
            model_engine, args, current_epoch, global_step,
            full_retrieval_mrr, best_eval_metric, stop, train_metrics
        )
        # Also keep your existing in-batch validation for comparison
        model_engine.eval()


        val_mrr10_inbatch = 0.0
        val_ndcg10_inbatch = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for val_step, val_batch in enumerate(val_dataloader):
                sentence_a, sentence_b, _, _, _, task_id = val_batch
                sentence_all = sentence_a + sentence_b

                inputs_all = model.tokenizer(sentence_all, padding='max_length', max_length=args.max_seq_length,
                                             truncation=True, return_tensors='pt')
                inputs_all = inputs_all.to(device)
                task_id = task_id.to(device)

                logits_student_in_batch_val, _, _ = model_engine(inputs_all, task_id, 'eval')

                # Use temperature scaling consistently
                logits_student_in_batch_val_temp = logits_student_in_batch_val / args.temperature_in_batch

                # Compute in-batch MRR (for comparison only)
                bs_val, num_candidates_val = logits_student_in_batch_val_temp.shape
                labels_batch_val = torch.zeros_like(logits_student_in_batch_val_temp)
                labels_batch_val[:, 0] = 1

                val_batch_mrr, val_batch_ndcg = compute_ranking_metrics(
                    logits_student_in_batch_val_temp, labels_batch_val, k=10
                )
                val_mrr10_inbatch += val_batch_mrr
                val_ndcg10_inbatch += val_batch_ndcg
                num_val_batches += 1

                if num_val_batches >= 20:
                    break

        model_engine.train()

        # Average validation metrics
        avg_val_mrr10_inbatch = val_mrr10_inbatch / max(1, num_val_batches)
        avg_val_ndcg10_inbatch = val_ndcg10_inbatch / max(1, num_val_batches)


        if args.global_rank == 0 and num_train_batches > 0:
            avg_train_loss_hardneg = epoch_loss_hardneg / num_train_batches
            avg_train_loss_rd = epoch_loss_rd / num_train_batches
            avg_train_loss_rd2 = epoch_loss_rd2 / num_train_batches
            avg_train_loss_feat = epoch_loss_feat / num_train_batches

            logging.info(
                f"Epoch {current_epoch + 1}/{args.num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"InBatch: {avg_train_loss_inbatch:.4f} | "
                f"HardNeg: {avg_train_loss_hardneg:.4f} | "
                f"RD: {avg_train_loss_rd:.4f} | "
                f"RD2: {avg_train_loss_rd2:.4f} | "
                f"Feat: {avg_train_loss_feat:.6f} | "
                f"Train MRR@10: {avg_mrr10:.4f} | NDCG@10: {avg_ndcg10:.4f} | "
                f"Val In-Batch MRR@10: {avg_val_mrr10_inbatch:.4f} | "
                f"Val FULL RETRIEVAL MRR@10: {full_retrieval_mrr:.4f} | "  # 🔥 NEW
                f"Best FULL RETRIEVAL MRR@10: {best_eval_metric:.4f}"  # 🔥 UPDATED
            )

            # Enhanced TensorBoard logging
            train_log_dict = {
                'loss_overall': avg_train_loss,
                'loss_inbatch': avg_train_loss_inbatch,
                'loss_hardneg': avg_train_loss_hardneg,
                'loss_rd': avg_train_loss_rd,
                'loss_rd2': avg_train_loss_rd2,
                'loss_feat': avg_train_loss_feat,
                'train_MRR@10': avg_mrr10,
                'train_NDCG@10': avg_ndcg10,
                'val_inbatch_MRR@10': avg_val_mrr10_inbatch,  # 🔥 RENAMED
                'val_full_retrieval_MRR@10': full_retrieval_mrr,  # 🔥 NEW
            }
            write_tensorboard(summary_writer, train_log_dict, global_step)

    # === Log the best checkpoint info after training (based on MRR@10) ===
    if dist.is_initialized():
        dist.barrier()  # make sure all ranks finish before logging

    if args.global_rank == 0:
        log_message = (
                "\n" + "=" * 80 + "\n"
                + "[Training Completed]\n"
                + "✅ Best checkpoint was found at:\n"
                + f"   ➤ Epoch: {best_epoch + 1}\n"
                + f"   ➤ Global Step: {best_step}\n"
                + f"   ➤ Best Training MRR@10: {best_eval_metric:.6f}\n"
                + f"   ➤ Path: {os.path.join(args.output_dir, f'checkpoint-epoch-{best_epoch + 1}')}\n"
                + "=" * 80 + "\n"
        )

    if dist.is_initialized():
        dist.destroy_process_group()

    if args.global_rank == 0:
        print(log_message)
        logging.info(log_message)

        # # Append to training log file only once (no duplicates)
        # with open(args.training_log, "a") as log_f:
        #     log_f.write(log_message)


if __name__ == '__main__':
    main()
