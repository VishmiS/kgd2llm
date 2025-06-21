# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.ln0 = nn.LayerNorm(dim_V) if ln else None
        self.ln1 = nn.LayerNorm(dim_V) if ln else None

    def forward(self, Q, K, pad_mask=None):
        Q_ = self.fc_q(Q)
        K_ = self.fc_k(K)
        V_ = self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q_.split(dim_split, 2), 0)
        K_ = torch.cat(K_.split(dim_split, 2), 0)
        V_ = torch.cat(V_.split(dim_split, 2), 0)

        pad_mask = pad_mask.unsqueeze(1).repeat(self.num_heads, Q.size(1), 1)
        score = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        score = score.masked_fill(pad_mask == 0, -1e12)
        A = torch.softmax(score, 2)
        A = A * pad_mask
        O = torch.cat(A.bmm(V_).split(Q.size(0), 0), 2)

        O = Q + O
        O = self.ln0(O) if self.ln0 else O
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O) if self.ln1 else O
        return O

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, pad_mask):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, pad_mask)

class IEM(nn.Module):
    def __init__(self, d_model, hidden, d_output, drop_prob=0.0):
        super(IEM, self).__init__()
        self.linear1 = nn.Linear(2 * d_model, hidden)
        self.proj0 = nn.Linear(hidden, hidden)
        self.proj1 = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, d_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, emb_a, emb_b):
        x = torch.cat((emb_a, emb_b), dim=-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x0 = self.relu(self.proj0(x))
        x1 = self.relu(self.proj1(x))
        rep = torch.stack((x0, x1), dim=0)
        logits = torch.cat((self.linear2(x0), self.linear2(x1)), dim=-1)
        return logits, rep

class Mymodel(nn.Module):
    def __init__(self, model_name_or_path, args):
        super(Mymodel, self).__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, pad_token='<|endoftext|>')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.plm_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.emb_dim = self.plm_model.transformer.wte.weight.size(1)
        self.mha_pma = PMA(self.emb_dim, args.num_heads, 1, ln=args.ln)
        self.iem = IEM(self.emb_dim, args.hidden_dim, args.output_dim)
        self.keep_max_layer = -1  # use last hidden layer

    def get_sentence_embedding(self, **inputs):
        outputs = self.plm_model(**inputs, output_hidden_states=True)
        embedding = outputs.hidden_states[self.keep_max_layer]
        mask = inputs['attention_mask']
        pooled = self.mha_pma(embedding, mask).squeeze(1)
        return F.normalize(pooled, p=2.0, dim=-1) if self.args.norm else pooled

    def encode(self, sentences, batch_size=64, convert_to_numpy=True):
        input_is_string = isinstance(sentences, str)
        if input_is_string:
            sentences = [sentences]

        embeddings = []
        with torch.no_grad():
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt', max_length=self.args.max_seq_length).to(self.plm_model.device)
                emb = self.get_sentence_embedding(**inputs)
                embeddings.extend(emb.cpu().numpy() if convert_to_numpy else emb)

        return embeddings[0] if input_is_string else embeddings
