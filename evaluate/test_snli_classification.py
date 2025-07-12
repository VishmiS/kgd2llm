import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from model.pro_model import Mymodel  # Your base model import


class MymodelClassifier(Mymodel):
    def __init__(self, *args, batch_size=32, device='cpu', **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_dim = 768  # Assuming BERT base embedding size; adjust if different
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 3)
        )
        self.batch_size = batch_size
        self.device = device

    def forward(self, premise_sentences, hypothesis_sentences):
        all_logits = []
        # Process in batches
        for i in range(0, len(premise_sentences), self.batch_size):
            batch_premises = premise_sentences[i:i + self.batch_size]
            batch_hypos = hypothesis_sentences[i:i + self.batch_size]

            # Encode and move embeddings to device
            premise_emb = self.encode(batch_premises, convert_to_tensor=True).to(self.device)
            hypo_emb = self.encode(batch_hypos, convert_to_tensor=True).to(self.device)

            combined = torch.cat([premise_emb, hypo_emb], dim=1)
            logits = self.classifier(combined)
            all_logits.append(logits)

        return torch.cat(all_logits, dim=0)


def load_model(checkpoint_path, device='cpu'):
    args = type('Args', (), {})()
    args.padding_side = 'right'
    args.num_heads = 8
    args.ln = True
    args.norm = True
    args.neg_K = 4

    model = MymodelClassifier(
        model_name_or_path='bert-base-uncased',
        max_seq_length=256,
        args=args,
        batch_size=32,
        device=device
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)

    # If your checkpoint contains additional keys (like 'model_state_dict'), update this accordingly:
    # e.g. state_dict = state_dict['model_state_dict']

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def run_inference_and_metrics(model, snli_tsv_path, device='cpu'):
    df = pd.read_csv(snli_tsv_path, sep='\t')

    premises = df['sentence1'].tolist()
    hypotheses = df['sentence2'].tolist()

    label_map = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    true_labels = df['gold_label'].map(label_map).tolist()

    with torch.no_grad():
        logits = model(premises, hypotheses)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().tolist()

    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, average='macro', zero_division=0)
    rec = recall_score(true_labels, preds, average='macro', zero_division=0)

    true_binarized = label_binarize(true_labels, classes=[0, 1, 2])
    ap = average_precision_score(true_binarized, probs.cpu().numpy(), average='macro')

    print(f"Accuracy: {acc:.4f}")
    print(f"Average Precision (AP): {ap:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "../PATH_TO_OUTPUT_MODEL/snli/final_student_model_fp32/pytorch_model.bin"
    snli_test_path = "../dataset/snli_1.0/snli_1.0_test.tsv"

    model = load_model(checkpoint_path, device)
    run_inference_and_metrics(model, snli_test_path, device)
