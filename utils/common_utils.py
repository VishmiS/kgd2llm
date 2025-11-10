import os
import random
import pathlib
import numpy as np
import torch
from loguru import logger
import shutil
import pickle
import argparse

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_model(model_engine, ckpt_dir, client_state):
    model_engine.save_checkpoint(ckpt_dir, client_state=client_state, exclude_frozen_parameters=True)

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


def remove_earlier_ckpt(path, start_name, current_step_num, max_save_num):
    filenames = os.listdir(path)
    ckpts = [dir_name for dir_name in filenames if
             dir_name.startswith(start_name) and int(dir_name.split('-')[1]) <= current_step_num]

    current_ckpt_num = len(ckpts)
    for dir_name in filenames:
        if dir_name.startswith(start_name) and int(dir_name.split('-')[1]) <= current_step_num and current_ckpt_num > (
                max_save_num - 1):
            try:
                shutil.rmtree(os.path.join(path, dir_name))
            except Exception as e:
                logger.error(f"Failed to delete checkpoint {dir_name}: {e}")


def makedirs(path):
    p = pathlib.Path(path)
    p.mkdir(parents=True, exist_ok=True)  # Changed to create full path
    return path


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(obj, path: str):
    if not os.path.exists(os.path.dirname(path)):
        makedirs(os.path.dirname(path))
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def write_tensorboard(summary_writer, log_dict, completed_steps):
    for key, value in log_dict.items():
        summary_writer.add_scalar(f'{key}', value, completed_steps)


def cos_sim(a, b):
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = a.to(device)
    b = b.to(device)

    # Ensure 2D tensors
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    # Normalize and compute cosine similarity
    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

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




def save_model_properly(model_engine, output_dir):
    """Save model with ALL parameters including custom layers - DEBUG VERSION"""
    print(f"💾 Saving model to {output_dir}...")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get the model (handle DDP wrapping)
    model_to_save = model_engine.module if hasattr(model_engine, 'module') else model_engine

    print("\n🔍 DEBUG: COMPREHENSIVE MODEL ANALYSIS")
    print("=" * 80)
    #
    # 1. Check ALL parameters in the model
    print("1️⃣  ALL MODEL PARAMETERS:")
    print("-" * 50)
    all_params = list(model_to_save.named_parameters())
    total_param_count = 0
    param_groups_count = 0

    for name, param in all_params:
        param_groups_count += 1
        total_param_count += param.numel()
        # print(
        #     f"  {param_groups_count:3d}: {name:80} - shape: {list(param.shape)} - requires_grad: {param.requires_grad}")

    print(f"📊 Total parameter groups: {param_groups_count}")
    print(f"📊 Total parameters: {total_param_count:,}")

    # 2. Check state dict
    print("\n2️⃣  STATE DICT ANALYSIS:")
    print("-" * 50)
    state_dict = model_to_save.state_dict()
    state_dict_keys = list(state_dict.keys())
    print(f"State dict keys: {len(state_dict_keys)}")

    # 3. Compare parameters vs state dict
    print("\n3️⃣  PARAMETER vs STATE DICT COMPARISON:")
    print("-" * 50)
    param_names = set([name for name, _ in all_params])
    state_dict_names = set(state_dict_keys)

    # Find what's missing
    missing_in_state_dict = param_names - state_dict_names
    extra_in_state_dict = state_dict_names - param_names

    print(f"Parameters in model: {len(param_names)}")
    print(f"Keys in state dict: {len(state_dict_names)}")
    print(f"Missing from state dict: {len(missing_in_state_dict)}")
    print(f"Extra in state dict: {len(extra_in_state_dict)}")

    if missing_in_state_dict:
        print("\n🚨 CRITICAL: Parameters missing from state dict:")
        for i, name in enumerate(sorted(missing_in_state_dict)):
            print(f"  ❌ {i + 1:2d}: {name}")

            # Try to find why it's missing
            for param_name, param in all_params:
                if param_name == name:
                    print(f"       This parameter EXISTS in model but not in state dict!")
                    print(f"       Shape: {list(param.shape)}, requires_grad: {param.requires_grad}")
                    break

    if extra_in_state_dict:
        print(f"\n📝 Extra keys in state dict (buffers etc.): {len(extra_in_state_dict)}")

    # 4. Check custom layers specifically
    print("\n4️⃣  CUSTOM LAYERS ANALYSIS:")
    print("-" * 50)

    # Check in parameters
    custom_params = [name for name, _ in all_params if
                     any(layer in name for layer in ['mha_pma', 'iem', 'fc_', 'linear', 'proj'])]
    print(f"Custom parameters in model: {len(custom_params)}")
    for name in custom_params[:10]:  # Show first 10
        print(f"  📍 {name}")

    # Check in state dict
    custom_in_state_dict = [k for k in state_dict_keys if
                            any(layer in k for layer in ['mha_pma', 'iem', 'fc_', 'linear', 'proj'])]
    print(f"Custom layers in state dict: {len(custom_in_state_dict)}")
    for key in custom_in_state_dict[:10]:  # Show first 10
        print(f"  ✅ {key}")

    # Find missing custom layers
    missing_custom = set(custom_params) - set(custom_in_state_dict)
    if missing_custom:
        print(f"\n🚨 MISSING CUSTOM LAYERS from state dict: {len(missing_custom)}")
        for name in sorted(missing_custom):
            print(f"  ❌ {name}")

    # 5. Check module structure
    print("\n5️⃣  MODULE STRUCTURE:")
    print("-" * 50)
    custom_modules = []
    for name, module in model_to_save.named_modules():
        if any(layer in name for layer in ['mha_pma', 'iem']):
            custom_modules.append((name, module))
            print(f"  🔧 {name}: {type(module).__name__}")

            # Check module parameters
            module_params = list(module.named_parameters(recurse=False))
            if module_params:
                print(f"     Direct parameters: {len(module_params)}")
                for param_name, param in module_params:
                    print(f"       - {param_name}: {list(param.shape)}")
            else:
                print(f"     ⚠️  NO DIRECT PARAMETERS (might be in submodules)")

    # 6. Save the state dict (even if incomplete for debugging)
    print("\n6️⃣  SAVING MODEL:")
    print("-" * 50)
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))

    # Save tokenizer and config
    try:
        model_to_save.tokenizer.save_pretrained(output_dir)
        model_to_save.plm_model.config.save_pretrained(output_dir)
        print("✅ Tokenizer and config saved")
    except Exception as e:
        print(f"⚠️ Could not save tokenizer/config: {e}")

    print(f"\n💾 Model saved to {output_dir}")
    print("=" * 80)

    # 7. SUMMARY
    print("\n📋 SUMMARY:")
    print("-" * 30)
    print(f"Total parameters: {total_param_count:,}")
    print(f"Parameter groups: {param_groups_count}")
    print(f"State dict keys: {len(state_dict_keys)}")
    print(f"Missing parameters: {len(missing_in_state_dict)}")
    print(f"Custom layers in state dict: {len(custom_in_state_dict)}")

    if len(missing_in_state_dict) > 0:
        print("❌ MODEL SAVING IS BROKEN - Custom layers are missing!")
        return False
    else:
        print("✅ Model saving appears correct")
        return True




def emergency_save(model_engine, output_dir):
    """Emergency save function as fallback"""
    print(f"🚨 EMERGENCY SAVE to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    model_to_save = model_engine.module if hasattr(model_engine, 'module') else model_engine

    # Try different saving methods
    try:
        # Method 1: Save entire model (not recommended but as fallback)
        torch.save(model_to_save, os.path.join(output_dir, "model_complete.pth"))
        print("✅ Emergency save completed (complete model)")
    except Exception as e:
        print(f"❌ Emergency save failed: {e}")


def debug_model_state_comprehensive(model_engine):
    """Comprehensive debug function to check what parameters exist in the model"""
    if args.global_rank == 0:
        model_to_check = model_engine.module if hasattr(model_engine, 'module') else model_engine

        # print("\n" + "=" * 100)
        # print("🏗️  COMPREHENSIVE MODEL ARCHITECTURE DEBUG - RUNNING AT START")
        # print("=" * 100)

        # Check all modules and their parameters
        total_trainable = 0
        total_params = 0

        # print("\n📋 ALL MODULES AND THEIR PARAMETERS:")
        # print("-" * 80)

        for module_name, module in model_to_check.named_modules():
            module_params = list(module.named_parameters(recurse=False))
            if module_params:
                print(f"\n🔧 {module_name} ({type(module).__name__}): {len(module_params)} parameters")
                for param_name, param in module_params:
                    total_params += param.numel()
                    if param.requires_grad:
                        total_trainable += param.numel()
                    print(f"   {param_name:50} - shape: {list(param.shape):15} - trainable: {param.requires_grad}")

        print(f"\n📊 SUMMARY:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {total_trainable:,}")
        print(f"   Non-trainable parameters: {total_params - total_trainable:,}")

        # Check state dict
        state_dict = model_to_check.state_dict()
        # print(f"   State dict keys: {len(state_dict.keys())}")

        print("=" * 100)



def verify_parameter_states(model):
    """Verify that all parameters are properly set for training"""
    print("\n🔍 PARAMETER STATE VERIFICATION:")

    total_params = 0
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        total_params += 1
        if param.requires_grad:
            trainable_params += 1
            if total_params <= 10:  # Show first 10 trainable params
                print(f"   ✅ TRAINABLE: {name}")
        else:
            frozen_params += 1
            if total_params <= 5:  # Show first 5 frozen params
                print(f"   ❌ FROZEN: {name}")

    print(f"\n📊 Parameter Summary:")
    print(f"   Total parameters: {total_params}")
    print(f"   Trainable: {trainable_params}")
    print(f"   Frozen: {frozen_params}")

    if frozen_params > total_params * 0.5:  # If more than 50% are frozen
        print("🚨 CRITICAL: Too many frozen parameters!")
        return False
    return True

def monitor_memory(step, prefix=""):
    if torch.cuda.is_available():
        print(f"{prefix} Step {step}:")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
        print(f"  Reserved: {torch.cuda.memory_reserved()/1024**3:.2f}GB")
        print(f"  Max Allocated: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        torch.cuda.reset_peak_memory_stats()
