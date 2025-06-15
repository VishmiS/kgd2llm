import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertModel, BertForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from transformers.modeling_outputs import SequenceClassifierOutput

# Load the BERT tokenizer and model (ensure correct vocab size)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
teacher_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
student_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')


# Helper class for PMA (Perceiver Multihead Attention)
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

    def forward(self, Q, K, V, pad_mask=None):
        # Linear projections
        Q = self.fc_q(Q)
        K = self.fc_k(K)
        V = self.fc_v(V)

        # Compute attention scores (Q * K^T) / sqrt(d_k)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)

        # Apply padding mask if given
        if pad_mask is not None:
            attention_scores = attention_scores.masked_fill(pad_mask == 0, -1e9)

        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

        # Apply attention weights to values (V)
        output = torch.matmul(attention_weights, V)

        # Final output projection
        output = self.fc_o(output)

        if hasattr(self, 'ln0'):
            output = self.ln0(output)
        return output


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))  # Seed vectors
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, pad_mask):
        # Repeat the seed tensor for each batch
        S = self.S.repeat(X.size(0), 1, 1)  # Repeat across batch dimension
        # Apply MAB with the seed tensor and the input tensor
        return self.mab(S, X, X, pad_mask)


# IEM Module
class IEM(nn.Module):
    def __init__(self, d_model, hidden, d_output, drop_prob=0.0):
        super(IEM, self).__init__()
        self.linear1 = nn.Linear(4 * d_model, hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, d_output)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, emb_a, emb_b):
        x = torch.cat((emb_a, emb_b), dim=-1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.proj(x)
        x = self.relu(x)
        logits = self.linear2(x)
        return logits, x


# Updated LoRA Student Model with PMA & IEM (BERT version)
class LoRAStudentModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', num_heads=8, num_seeds=1, d_model=768, hidden=512,
                 drop_prob=0.1, num_classes=3):
        super(LoRAStudentModel, self).__init__()

        # Load pretrained BERT model (without classifier head)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        # print(f"Student vocab size: {self.model.config.vocab_size}")

        # Your PMA and IEM modules
        self.pma = PMA(d_model, num_heads, num_seeds)
        self.iem = IEM(d_model, hidden, d_output=num_classes, drop_prob=drop_prob)

        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, peft_config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get embeddings from last hidden layer (batch_size, seq_len, hidden_dim)
        outputs = self.model.bert(input_ids=input_ids, attention_mask=attention_mask,
                                  return_dict=True, output_hidden_states=True)
        embeddings = outputs.hidden_states[-1]  # Last layer embeddings

        cls_embedding = embeddings[:, 0, :]  # (batch_size, d_model)
        pma_output = self.pma(embeddings, attention_mask)  # (batch_size, seq_len, d_model)
        pma_pooled = pma_output.mean(dim=1)  # (batch_size, d_model)

        # Concatenate CLS and PMA pooled
        combined_emb = torch.cat([cls_embedding, pma_pooled], dim=-1)  # (batch_size, 2 * d_model)

        logits, rep = self.iem(combined_emb, combined_emb)

        # If logits have a sequence dimension, average it
        if logits.dim() == 3:
            logits = logits.mean(dim=1)  # shape: [batch_size, num_classes]

        return logits, rep, cls_embedding