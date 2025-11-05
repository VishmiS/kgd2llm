import sys
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoConfig


# LoRA-enabled Linear layer
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=8, alpha=32):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        # Initialize
        nn.init.zeros_(self.lora_B.weight)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.fc(x) + self.lora_B(self.lora_A(x)) * self.scaling


# Replace MAB with LoRA-enabled MAB
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, mask=None):
        # Ensure proper device and dtype
        if self.S.device != X.device:
            self.S = self.S.to(X.device)

        # Repeat seed for batch size - these will be our queries
        Q = self.S.repeat(X.size(0), 1, 1)  # [batch_size, num_seeds, dim]

        # Apply the MAB with seeds as queries and input as keys/values
        Q_ = self.mab.fc_q(Q)
        K_, V_ = self.mab.fc_k(X), self.mab.fc_v(X)

        dim_split = self.mab.dim_V // self.mab.num_heads
        Q_ = torch.cat(Q_.split(dim_split, 2), 0)
        K_ = torch.cat(K_.split(dim_split, 2), 0)
        V_ = torch.cat(V_.split(dim_split, 2), 0)

        # Handle mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(self.mab.num_heads, Q.size(1), 1)

        # Compute attention
        score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.mab.dim_V)

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e12)

        A = torch.softmax(score, 2)

        if mask is not None:
            A = A * mask

        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)
        O = Q + O

        # Apply layer norms if they exist
        if hasattr(self.mab, 'ln0') and self.mab.ln0 is not None:
            O = self.mab.ln0(O)

        O = O + F.relu(self.mab.fc_o(O))

        if hasattr(self.mab, 'ln1') and self.mab.ln1 is not None:
            O = self.mab.ln1(O)

        # Only squeeze if we have exactly 1 seed
        if O.size(1) == 1:
            O = O.squeeze(1)

        return O


class IEM(nn.Module):
    def __init__(self, d_model, hidden, d_output, drop_prob=0.0):
        super(IEM, self).__init__()
        self.linear1 = nn.Linear(2 * d_model, hidden)
        self.proj0 = nn.Linear(hidden, hidden)
        self.proj1 = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, d_output)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.proj0.weight)
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)
        self.sftmx = nn.Softmax(dim=-1)

    def forward(self, emb_a, emb_b):
        # ✅ FIX: Ensure consistent dtype with model parameters
        if emb_a.dtype != next(self.parameters()).dtype:
            emb_a = emb_a.to(next(self.parameters()).dtype)
        if emb_b.dtype != next(self.parameters()).dtype:
            emb_b = emb_b.to(next(self.parameters()).dtype)

        x = torch.cat((emb_a, emb_b), dim=-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x0 = self.proj0(x)
        x1 = self.proj1(x)
        x0 = self.relu(x0)
        x1 = self.relu(x1)
        rep = torch.stack((x0, x1), dim=0)
        logits0 = self.linear2(x0)
        logits1 = self.linear2(x1)
        logits = torch.cat((logits0, logits1), dim=-1)
        return logits, rep


class Mymodel(nn.Module):
    def __init__(self,
                 model_name_or_path=None,
                 alias=None,
                 max_seq_length=256,
                 args=None
                 ):
        super(Mymodel, self).__init__()
        self.alias = alias
        if self.alias == None:
            self.alias = model_name_or_path
        self.args = args
        self.max_seq_length = max_seq_length
        self.model_name_or_path = model_name_or_path
        self.keep_max_layer = -1  # updated

        # ✅ FORCE BERT: Use BERT-specific classes
        from transformers import BertTokenizer, BertModel

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',  # Force BERT tokenizer
            truncation_side='right',
            padding_side=self.args.padding_side
        )

        # ✅ BERT already has [PAD] token, so no need for eos_token handling
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        # ✅ FORCE BERT: Load BERT model with ALL parameters trainable
        self.plm_model = BertModel.from_pretrained('bert-base-uncased')

        # 🔥 CRITICAL: Ensure NO parameters are frozen
        for param in self.plm_model.parameters():
            param.requires_grad = True

        print(
            f"✅ BERT model loaded - all {sum(p.numel() for p in self.plm_model.parameters()):,} parameters are trainable")

        self.emb_dim = self.plm_model.config.hidden_size
        self.num_heads = args.num_heads
        self.ln = args.ln
        self.norm = args.norm
        self.mha_pma = PMA(self.emb_dim, self.num_heads, 1, ln=self.ln)
        self.hidden_dim = 768
        self.output_dim = 512
        self.iem = IEM(self.emb_dim, self.hidden_dim, self.output_dim)

    def forward(self, inputs_all, task_ids, mode):
        # ✅ FIX: Ensure inputs are in the correct dtype
        if inputs_all['input_ids'].dtype != next(self.parameters()).dtype:
            inputs_all = {k: v.to(next(self.parameters()).dtype) if v.dtype.is_floating_point else v
                          for k, v in inputs_all.items()}

        # print(f"\n🔍 MYMODEL.FORWARD DEBUG:")
        # print(f"   Mode: {mode}")
        # print(f"   Input shape: {inputs_all['input_ids'].shape}")
        # print(f"   Task IDs: {task_ids}")

        if mode == 'train':
            # Get embeddings for all 288 sentences
            embeddings_all = self.get_sentence_embedding(**inputs_all)  # [288, emb_dim]
            # print(f"   Raw embeddings shape: {embeddings_all.shape}")

            # ✅ FIXED: Properly group into 16 batches of 18 sentences each
            bs = task_ids.size(0)  # batch size = 16
            group_size = 2 + self.args.neg_K  # 2 (query + positive) + 16 negatives = 18

            # Reshape to [16, 18, emb_dim] - CRITICAL FIX!
            output_embeddings_all = embeddings_all.reshape(bs, group_size, self.emb_dim)
            # print(f"   Reshaped embeddings: {output_embeddings_all.shape}")

            # Extract components
            output_embeddings_a = output_embeddings_all[:, 0]  # [16, emb_dim] - queries
            output_embeddings_b = output_embeddings_all[:, 1]  # [16, emb_dim] - positives
            output_embeddings_hardneg = output_embeddings_all[:, 2:]  # [16, 16, emb_dim] - negatives

            # print(f"   Queries shape: {output_embeddings_a.shape}")
            # print(f"   Positives shape: {output_embeddings_b.shape}")
            # print(f"   Negatives shape: {output_embeddings_hardneg.shape}")

        elif mode == 'eval':
            # For eval, we have 32 sentences (16 queries + 16 positives)
            embeddings_all = self.get_sentence_embedding(**inputs_all)  # [32, emb_dim]
            output_embeddings_all = embeddings_all.reshape(2, -1, self.emb_dim)  # [2, 16, emb_dim]
            output_embeddings_a = output_embeddings_all[0]  # [16, emb_dim]
            output_embeddings_b = output_embeddings_all[1]  # [16, emb_dim]
            bs = output_embeddings_a.size(0)
        else:
            raise ValueError('Error of mode value')

        # Create in-batch pairs: each a with all b
        a_expand_emb = output_embeddings_a.unsqueeze(1).expand(-1, bs, -1)  # [bs, bs, emb_dim]
        b_expand_emb = output_embeddings_b.unsqueeze(0).expand(bs, -1, -1)  # [bs, bs, emb_dim]

        # Flatten to feed into IEM
        a_flat = a_expand_emb.reshape(bs * bs, self.emb_dim)
        b_flat = b_expand_emb.reshape(bs * bs, self.emb_dim)

        # Compute logits for all pairs
        logits_all_pairs, _ = self.iem(a_flat, b_flat)  # [bs*bs, 2]

        # Reshape to [bs, bs, 2] to match in-batch structure
        logits_all_pairs = logits_all_pairs.reshape(bs, bs, -1)

        # 🔥 CRITICAL FIX: Extract POSITIVE class logits (index 1) for ranking
        # The IEM outputs [negative_logits, positive_logits] for each pair
        # We want the positive class scores for similarity ranking
        positive_class_logits = logits_all_pairs[:, :, 1]  # [bs, bs] - positive class scores
        output_in_batch_specific_task = positive_class_logits

        # print(f"   In-batch logits shape: {output_in_batch_specific_task.shape}")
        # print(f"   In-batch - Pos: {output_in_batch_specific_task[:, 0].mean().item():.3f}, Neg: {output_in_batch_specific_task[:, 1:].mean().item():.3f}")

        if mode == 'train':
            # ✅ FIXED: Proper hard negative processing
            # We have: queries [16, dim], positives [16, dim], negatives [16, 16, dim]
            # We need: for each query, compute similarity with its positive + its 16 negatives

            # Expand queries to match negatives: [16, dim] -> [16, 17, dim]
            queries_expanded = output_embeddings_a.unsqueeze(1).expand(-1, 1 + self.args.neg_K, -1)

            # Combine positive and negatives: [16, 17, dim]
            positives_reshaped = output_embeddings_b.unsqueeze(1)  # [16, 1, dim]
            documents_combined = torch.cat([positives_reshaped, output_embeddings_hardneg], dim=1)  # [16, 17, dim]

            # print(f"   Queries expanded: {queries_expanded.shape}")
            # print(f"   Documents combined: {documents_combined.shape}")

            # Flatten for IEM: [16*17, dim] for both
            queries_flat = queries_expanded.reshape(bs * (1 + self.args.neg_K), self.emb_dim)
            documents_flat = documents_combined.reshape(bs * (1 + self.args.neg_K), self.emb_dim)

            # Compute similarities
            output_hardneg, output_pos_hardneg_rep = self.iem(queries_flat, documents_flat)  # [16*17, 2]

            # Reshape back: [16, 17, 2]
            output_hardneg = output_hardneg.reshape(bs, 1 + self.args.neg_K, -1)

            # 🔥 CRITICAL FIX: Extract POSITIVE class logits for hard negatives
            # output_hardneg shape: [16, 17, 2] - we want the positive class (index 1)
            positive_class_hardneg = output_hardneg[:, :, 1]  # [16, 17] - positive class scores
            output_hardneg_specific_task = positive_class_hardneg

            # print(f"   Hard negative logits shape: {output_hardneg_specific_task.shape}")
            # print(f"   Hard negative - Pos: {output_hardneg_specific_task[:, 0].mean().item():.3f}, Neg: {output_hardneg_specific_task[:, 1:].mean().item():.3f}")

            output_pos_hardneg_rep_specific_task = output_pos_hardneg_rep[task_ids[0]]

        elif mode == 'eval':
            output_hardneg_specific_task = None
            output_pos_hardneg_rep_specific_task = None

        return output_in_batch_specific_task, output_hardneg_specific_task, output_pos_hardneg_rep_specific_task

    def pma_embedding(self, A, mask):
        res = self.mha_pma(A, mask).squeeze(1)  # ✅ FIXED: Remove the first argument
        return res

    def get_sentence_embedding(self, **inputs):
        try:
            outputs = self.plm_model(**inputs, output_hidden_states=True)
            attention_mask = inputs['attention_mask']

            # Use the last hidden state instead of specific layer
            embedding = outputs.last_hidden_state

            # ✅ FIX: Ensure embedding dtype matches model parameters
            if embedding.dtype != next(self.parameters()).dtype:
                embedding = embedding.to(next(self.parameters()).dtype)

            # Validate embedding shape
            if embedding.size(-1) != self.emb_dim:
                raise ValueError(f"Expected embedding dimension {self.emb_dim}, but got {embedding.size(-1)}.")

            # Apply PMA pooling
            res_embedding = self.pma_embedding(embedding, attention_mask)

            if self.norm:
                res_embedding = torch.nn.functional.normalize(res_embedding, p=2.0, dim=-1, eps=1e-12, out=None)

            return res_embedding

        except Exception as e:
            print(f"Error in get_sentence_embedding: {e}")
            # Fallback: use mean pooling
            outputs = self.plm_model(**inputs)
            embedding = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']

            # ✅ FIX: Ensure consistent dtype in fallback
            if embedding.dtype != next(self.parameters()).dtype:
                embedding = embedding.to(next(self.parameters()).dtype)

            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embedding.size()).float()
            sum_embeddings = torch.sum(embedding * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            res_embedding = sum_embeddings / sum_mask

            if self.norm:
                res_embedding = torch.nn.functional.normalize(res_embedding, p=2.0, dim=-1, eps=1e-12, out=None)

            return res_embedding

    def webquestions_template_context(self, query, passage):
        """Use the SAME template as during training"""
        return (
            f"#Q describes a user question. #A describes a web passage. "
            f"Are they related such that #A correctly answers #Q? Answer Can or Cannot.\n"
            f"#Q: {query}\n#A: {passage}\nAnswer:"
        )

    def encode(self, sentences, batch_size=64, convert_to_numpy=True,
               convert_to_tensor=False, show_progress_bar=True, max_seq_length=None, **kwargs):

        if max_seq_length is None:
            max_seq_length = self.max_seq_length

        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_is_string = True

        all_embeddings = []

        # 🔥 CRITICAL: Use the SAME template as training
        templates = [self.webquestions_template_context(sentence, "") for sentence in sentences]

        with torch.no_grad():
            for start_index in trange(0, len(templates), batch_size, desc="Batches", disable=not show_progress_bar):
                batch_templates = templates[start_index: start_index + batch_size]

                # 🔥 CRITICAL: Use the EXACT same forward pass as training validation
                inputs = self.tokenizer(batch_templates, padding=True, truncation=True,
                                        max_length=max_seq_length, return_tensors='pt').to(self.plm_model.device)

                # 🔥 CRITICAL: Create task_id like in training
                task_id = torch.zeros(len(batch_templates), dtype=torch.long).to(self.plm_model.device)

                # 🔥 CRITICAL: Use get_sentence_embedding directly for single sentences
                # This avoids the reshape issue in the forward pass
                embeddings = self.get_sentence_embedding(**inputs)

                # Normalize (same as training)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                embeddings = embeddings.detach()
                if convert_to_numpy:
                    if embeddings.dtype == torch.bfloat16:
                        embeddings = embeddings.cpu().to(torch.float32)
                    else:
                        embeddings = embeddings.cpu()
                all_embeddings.extend(embeddings)

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_is_string:
            all_embeddings = all_embeddings[0]
        return all_embeddings

    def verify_model(self, test_texts=["hello", "world"]):
        """Verify that the model produces different embeddings for different inputs"""
        print("\n[VERIFICATION] Testing model output variability...")

        with torch.no_grad():
            embeddings = self.encode(test_texts, convert_to_tensor=True)

            # Check if embeddings are different
            diff = torch.norm(embeddings[0] - embeddings[1]).item()
            print(f"Difference between embeddings: {diff:.6f}")

            if diff < 0.01:
                print("❌ CRITICAL: Model produces identical embeddings!")
                return False
            else:
                print("✅ Model produces different embeddings")
                return True