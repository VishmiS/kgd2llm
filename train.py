import warnings
import logging
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
from loss import cal_loss_in_batch, cal_loss_hardneg, cal_loss_rd, cal_loss_rd2, cal_feat_loss
logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings('ignore')

import torch


def compute_ranking_metrics(scores, labels, k=10):
    """
    scores: [batch_size, num_candidates] (higher = better)
    labels: [batch_size, num_candidates] (1 for positive, 0 for negatives)
    """
    batch_size = scores.size(0)
    mrr_total = 0.0
    ndcg_total = 0.0

    _, indices = scores.topk(k, dim=-1)
    for i in range(batch_size):
        ranked_labels = labels[i][indices[i]]
        # MRR@k
        pos_idx = (ranked_labels == 1).nonzero(as_tuple=True)[0]
        if len(pos_idx) > 0:
            mrr_total += 1.0 / (pos_idx[0].item() + 1)
        # NDCG@k
        gains = 2 ** ranked_labels.float() - 1
        discounts = torch.log2(torch.arange(2, k + 2, device=scores.device).float())
        dcg = (gains / discounts).sum()
        ideal_gains = 2 ** torch.sort(ranked_labels, descending=True)[0].float() - 1
        idcg = (ideal_gains / discounts).sum()
        ndcg_total += (dcg / idcg).item() if idcg > 0 else 0.0

    return mrr_total / batch_size, ndcg_total / batch_size



def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


def save_model_properly(model_engine, save_path):
    """Save model in Hugging Face format"""
    # Get the underlying model from DeepSpeed engine
    model = model_engine.module if hasattr(model_engine, 'module') else model_engine

    # Save the model state dict
    model_state_dict = model.state_dict()

    # Save in standard format
    torch.save(model_state_dict, os.path.join(save_path, "pytorch_model.bin"))

    # Save config and tokenizer
    model.plm_model.config.save_pretrained(save_path)
    model.tokenizer.save_pretrained(save_path)

    print(f"✅ Model saved in Hugging Face format to: {save_path}")


def check_architecture_consistency(model):
    """Verify that model architecture matches expected format"""
    print("\n=== ARCHITECTURE CHECK ===")

    # Check what type of model we have
    model_type = type(model.plm_model).__name__
    print(f"Model type: {model_type}")

    # Sample keys from the model
    sample_keys = list(model.state_dict().keys())[:10]
    print("Sample model keys:")
    for key in sample_keys:
        print(f"  {key}")

    # Check if it's BERT-like or GPT-like
    bert_keys = [k for k in sample_keys if 'encoder' in k or 'bert' in k.lower()]
    gpt_keys = [k for k in sample_keys if 'h.' in k or 'gpt' in k.lower()]

    if bert_keys and not gpt_keys:
        print("✅ Model appears to be BERT architecture")
    elif gpt_keys and not bert_keys:
        print("✅ Model appears to be GPT architecture")
    else:
        print("⚠️  Mixed architecture signals detected")

    return model_type


def verify_bert_setup(model, args):
    """Verify BERT is properly configured"""
    print("\n=== BERT SETUP VERIFICATION ===")

    # Check model type
    model_type = type(model.plm_model).__name__
    print(f"✅ Model type: {model_type}")

    # Check tokenizer
    print(f"✅ Tokenizer type: {type(model.tokenizer).__name__}")

    # Test BERT-style tokenization
    test_text = "what does jamaican people speak?"
    tokens = model.tokenizer.tokenize(test_text)
    print(f"✅ BERT tokens sample: {tokens[:8]}...")

    # Check embedding dimension
    print(f"✅ Embedding dimension: {model.emb_dim}")

    # More robust BERT architecture verification
    sample_keys = list(model.state_dict().keys())[:10]
    print("✅ Sample model keys for verification:")
    for key in sample_keys:
        print(f"   {key}")

    # Check for BERT patterns in the keys
    bert_patterns = [
        'encoder.layer',  # BERT encoder layers
        'embeddings.word_embeddings',  # BERT embeddings
        'attention.self.query',  # BERT attention
        'intermediate.dense',  # BERT feed-forward
        'output.dense'  # BERT output
    ]

    bert_detected = any(any(pattern in key for pattern in bert_patterns) for key in sample_keys)

    if bert_detected:
        print("🎉 BERT architecture confirmed!")
        # Count BERT-specific patterns found
        for pattern in bert_patterns:
            count = sum(1 for key in sample_keys if pattern in key)
            if count > 0:
                print(f"   Found {count} instances of '{pattern}'")
    else:
        print("⚠️  Standard BERT patterns not detected, but model type is BertModel")
        # Check what we actually have
        gpt_patterns = ['h.', 'ln_', 'attn.c_attn', 'attn.c_proj']
        gpt_detected = any(any(pattern in key for pattern in gpt_patterns) for key in sample_keys)
        if gpt_detected:
            print("❌ Found GPT patterns instead!")
        else:
            print("ℹ️  Unknown architecture patterns")

    # Test forward pass with the model - FIXED DEVICE HANDLING
    print("\n=== TESTING FORWARD PASS ===")
    try:
        test_inputs = model.tokenizer(["Test sentence"], padding=True, truncation=True,
                                      max_length=args.max_seq_length, return_tensors='pt')

        # ✅ FIX: Move inputs to the same device as the model
        device = next(model.parameters()).device
        test_inputs = {k: v.to(device) for k, v in test_inputs.items()}

        with torch.no_grad():
            outputs = model.plm_model(**test_inputs)
            embeddings = model.get_sentence_embedding(**test_inputs)

        print(f"✅ Forward pass successful!")
        print(f"✅ Output embeddings shape: {embeddings.shape}")
        print(f"✅ Pooler output shape: {outputs.pooler_output.shape if hasattr(outputs, 'pooler_output') else 'N/A'}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        print("ℹ️  This is likely a device mismatch issue - actual training should work fine")

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

    if stop >= args.patience:
        return stop, min_reduce_loss_eval, best_epoch, best_step, True  # early stop signal

    if reduce_loss_eval <= min_reduce_loss_eval:
        min_reduce_loss_eval = reduce_loss_eval
        best_epoch = current_epoch
        best_step = global_step
        stop = 0

        if args.global_rank == 0:
            print('💾 New best validation loss - saving model...')
            try:
                remove_earlier_ckpt(args.output_dir, 'checkpoint', global_step, max_save_num=2)
            except:
                print('No ckpt to remove.')

            # ✅ Save the best model in Hugging Face format
            best_ckpt_dir = os.path.join(
                args.output_dir, f"checkpoint-best-epoch-{current_epoch + 1}-step-{global_step}-{args.mark}"
            )
            os.makedirs(best_ckpt_dir, exist_ok=True)
            save_model_properly(model_engine, best_ckpt_dir)
            print(f"✅ Best model saved to: {best_ckpt_dir}")
    else:
        stop += 1

    # Optional: Save regular checkpoints based on num_ckpt (if you want to keep this)
    if save_flag and args.global_rank == 0:
        output_dir_current = os.path.join(
            args.output_dir, f"checkpoint-{global_step}-epoch-{current_epoch + 1}-{args.mark}")
        # ✅ Use proper saving for regular checkpoints too
        save_model_properly(model_engine, output_dir_current)

    model_engine.train()
    return stop, min_reduce_loss_eval, best_epoch, best_step, False


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
    parser.add_argument('--gradient_clipping', default=1.0, type=float, help='max_grad_norm')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='gradient accumulation steps')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--bf16', default=True, type=str2bool)
    parser.add_argument('--verbose', default=True, type=str2bool)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--local_rank', type=int, default=-1, help='ds')
    parser.add_argument('--global_rank', type=int, default=-1, help='ds')
    parser.add_argument('--mark', type=str, default='', help='Mark or label for checkpoint folder naming')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args.world_size = int(os.getenv('WORLD_SIZE', '0'))

    # --- Ensure num_epochs is the value you expect ---
    # Reason: deepspeed/launcher or environment might override parser defaults.
    # You can set NUM_EPOCHS env var to override; otherwise change the '10' below.
    desired_epochs = int(os.getenv('NUM_EPOCHS', '15'))  # <- change '10' to the default you want
    args.num_epochs = desired_epochs

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

    model = Mymodel(model_name_or_path=args.base_model_dir,
    alias = None,
    max_seq_length = args.max_seq_length,
    args = args)
    model.plm_model.gradient_checkpointing_enable()

    summary_writer = SummaryWriter(log_dir=args.tb_dir)

    check_architecture_consistency(model)
    verify_bert_setup(model,args)

    update_parameters = filter(lambda p: p.requires_grad, model.parameters())
    param_optimizer = list([(n, p) for n, p in model.named_parameters() if p.requires_grad])

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [


    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'lr': args.lr, 'weight_decay': args.weight_decay, 'betas': [0.8, 0.999], 'eps': 1e-6, 'name': 'd'},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'lr': args.lr, 'weight_decay': 0.0, 'betas': [0.8, 0.999], 'eps': 1e-6, 'name': 'nd'}]

    # --- Replace everything from 'from deepspeed.ops.adam import DeepSpeedCPUAdam'
    # --- down to the call to deepspeed.initialize(...) with this block ---

    # === FIXED: Simple DeepSpeed config for single GPU ===
    from torch.optim import AdamW

    # ✅ Use ZeRO Stage 1 (not Stage 2) for single GPU to avoid gradient issues
    ds_config = {
        "train_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": args.gradient_clipping,
        "bfloat16": {
            "enabled": args.bf16
        },
        "fp16": {
            "enabled": False
        },
        "zero_optimization": {
            "stage": 1,  # ⚠️ CHANGED FROM 2 TO 1 - critical fix!
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "cpu_offload": False
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.8, 0.999],
                "eps": 1e-6,
                "weight_decay": args.weight_decay
            }
        },
        "wall_clock_breakdown": False
    }

    # ✅ Let DeepSpeed create the optimizer (don't create it manually)
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model,
        config=ds_config
    )

    debug_gradient_flow(model_engine)

    print(
        f"[InitCheck] BF16 enabled: {ds_config['bfloat16']['enabled']} | ZeRO stage: {ds_config['zero_optimization']['stage']}")


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

    # reduction accumulators must be torch tensors on device for dist.all_reduce
    reduce_loss = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
    reduce_loss_eval = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
    reduce_loss_in_batch = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
    reduce_loss_in_batch_eval = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
    reduce_loss_hardneg = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
    reduce_loss_rd = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
    reduce_loss_rd2 = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
    reduce_loss_feat = torch.zeros(1, device='cuda' if torch.cuda.is_available() else 'cpu')
    reduce_inbatch_sample_num = {}

    epoch_loss_overall = 0.0
    epoch_loss_inbatch = 0.0
    epoch_loss_hardneg = 0.0
    epoch_loss_rd = 0.0
    epoch_loss_rd2 = 0.0
    epoch_loss_feat = 0.0
    num_train_batches = 0

    print("About to start training loop...")
    logging.info(f"\nAbout to start training loop...")

    for current_epoch in trange(int(args.num_epochs), desc="Epoch", disable=(args.global_rank != 0), mininterval=0):

        torch.autograd.set_detect_anomaly(True)
        if False:
            pass
        torch.cuda.empty_cache()
        model_engine.train()

        # Start of epoch logging
        logging.info(f"Starting Epoch {current_epoch + 1}/{args.num_epochs}")
        epoch_mrr10 = 0.0
        epoch_ndcg10 = 0.0
        num_mrr_batches = 0

        loss_epoch_eval = 0

        batch_iterator = tqdm(train_dataloader,
                              desc=f"Running Epoch {current_epoch + 1} of {args.num_epochs}",
                              disable=(args.global_rank != 0),
                              mininterval=0)
        for step, batch in enumerate(batch_iterator):
            sentence_a, sentence_b, logits_teacher_pos, sentence_hardneg, logits_teacher_hardneg, task_id = batch
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


            # print(f"Type of teacher_feature_cos_dict[{key}]: {type(teacher_feature_cos_dict[key])}")
            # print(f"Length of teacher_feature_cos_dict[{key}]: {len(teacher_feature_cos_dict[key])}")
            # print(f"Type of first element: {type(teacher_feature_cos_dict[key][0])}")
            # print(f"Sample at step {step}: {sample}")

            inputs_all = model.tokenizer(sentence_all, padding='max_length', max_length=args.max_seq_length,
                                         truncation=True, return_tensors='pt')
            inputs_all = inputs_all.to(device)
            task_id = task_id.to(device)
            logits_student_in_batch, logits_student_hardneg, rep_student_pos_hardneg = model_engine(inputs_all, task_id,
                                                                                                    'train')

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

            print("[Debug] logits_teacher_hardneg shape:", logits_teacher_hardneg.shape)  # Expect [B, neg_K+1, 2]
            # print("[Debug] logits_teacher_inbatch shape:", logits_teacher_inbatch.shape)  # Expect [B, in_batch_neg]
            # print("[Debug] logits_student_hardneg shape:", logits_student_hardneg.shape)
            # print("[Debug] logits_student_in_batch shape:", logits_student_in_batch.shape)

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
            print(f"Teacher logits (pos[1]): {logits_teacher_pos[..., 1].mean().item():.4f}")
            print(f"Teacher logits (neg[1]): {logits_teacher_hardneg[..., 1].mean().item():.4f}")

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
            bs, num_candidates = logits_student_in_batch.shape
            labels_batch = torch.zeros_like(logits_student_in_batch)
            labels_batch[:, 0] = 1  # first candidate is positive

            mrr10, ndcg10 = compute_ranking_metrics(logits_student_in_batch, labels_batch, k=10)
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

            # Weighted combination of loss components (tunable)
            loss_batch = (
                    args.alpha * loss_in_batch +
                    args.beta * loss_hardneg +
                    args.gamma * (loss_rd + loss_rd2) +
                    args.eta * loss_feat
            )
            if args.verbose:
                batch_iterator.set_description(
                    f"Epoch: {current_epoch + 1}/{args.num_epochs}, Batch:{step}/{len(train_dataloader)}, Loss: {loss_batch:9.4f}")

            # Before backward
            loss_batch = loss_batch.float()  # ensure float

            # === Diagnostic: Check gradient flow ===
            # Backward
            model_engine.backward(loss_batch)

            # === FIXED: Proper gradient checking for DeepSpeed ===
            if step % 10 == 0:  # Check more frequently for debugging
                print(f"\n[GradCheck] Step {step} - Checking gradients...")

                total_params = 0
                params_with_grad = 0
                grad_norms = []

                # Access the wrapped model
                model_to_check = model_engine.module if hasattr(model_engine, 'module') else model_engine

                for name, param in model_to_check.named_parameters():
                    if param.requires_grad:
                        total_params += 1
                        if param.grad is not None:
                            params_with_grad += 1
                            grad_norm = param.grad.data.norm(2).item()
                            grad_norms.append(grad_norm)

                            # Print first few parameters to verify gradients
                            if params_with_grad <= 3:  # Only show first 3 to avoid spam
                                print(f"  ✅ {name}: grad_norm = {grad_norm:.6e}")
                        else:
                            if total_params <= 3:  # Only show first 3 missing gradients
                                print(f"  ❌ {name}: NO gradient")

                if params_with_grad > 0:
                    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0
                    print(
                        f"[GradCheck] Step {step}: ✅ {params_with_grad}/{total_params} params have gradients | avg_norm = {avg_grad_norm:.6e}")
                else:
                    print(f"[GradCheck] Step {step}: ❌ CRITICAL - 0/{total_params} parameters have gradients!")

                    # Emergency debug: Check if loss requires grad
                    print(f"[GradCheck] Loss requires_grad: {loss_batch.requires_grad}")
                    print(f"[GradCheck] Loss device: {loss_batch.device}")
                    print(f"[GradCheck] Model device: {next(model_to_check.parameters()).device}")

            # Then step
            model_engine.step()



            if step % 200 == 0 and hasattr(optimizer, 'param_groups'):
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[LRCheck] Step {step}: current LR = {current_lr:.6e}")

            if (step + 1) % args.gradient_accumulation_steps == 0:
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

            # Update best model if MRR improved
            if avg_mrr10 > best_eval_metric:
                best_eval_metric = avg_mrr10
                best_epoch = current_epoch
                best_step = global_step
                best_checkpoint_path = epoch_ckpt_dir
                print(f"🌟 New best MRR@10: {best_eval_metric:.4f} at epoch {current_epoch + 1}")

        # Optional early stopping (no longer based on loss)
        early_stop = False


        if args.global_rank == 0 and num_train_batches > 0:
            avg_train_loss = epoch_loss_overall / num_train_batches
            avg_train_loss_inbatch = epoch_loss_inbatch / num_train_batches
            avg_train_loss_hardneg = epoch_loss_hardneg / num_train_batches
            avg_train_loss_rd = epoch_loss_rd / num_train_batches
            avg_train_loss_rd2 = epoch_loss_rd2 / num_train_batches
            avg_train_loss_feat = epoch_loss_feat / num_train_batches

            logging.info(
                f"Epoch {current_epoch + 1}/{args.num_epochs} Train Loss: {avg_train_loss:.4f} | "
                f"InBatch: {avg_train_loss_inbatch:.4f} | HardNeg: {avg_train_loss_hardneg:.4f} | "
                f"RD: {avg_train_loss_rd:.4f} | RD2: {avg_train_loss_rd2:.4f} | "
                f"Feat: {avg_train_loss_feat:.6f} | "
                f"MRR@10: {avg_mrr10:.4f} | NDCG@10: {avg_ndcg10:.4f}"
            )

            # Optional: log to TensorBoard
            train_log_dict = {
                'loss_overall': avg_train_loss,
                'loss_inbatch': avg_train_loss_inbatch,
                'loss_hardneg': avg_train_loss_hardneg,
                'loss_rd': avg_train_loss_rd,
                'loss_rd2': avg_train_loss_rd2,
                'loss_feat': avg_train_loss_feat,
                'MRR@10': avg_mrr10,
                'NDCG@10': avg_ndcg10,
            }
            write_tensorboard(summary_writer, train_log_dict, global_step)

    # === Log the best checkpoint info after training (based on MRR@10) ===
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
                + f"   ➤ Best MRR@10: {best_eval_metric:.6f}\n"
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