import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from model.pro_model import Mymodel  # Your base model import


class STSClassifier(Mymodel):
    def __init__(self, *args, batch_size=32, device='cpu', num_labels=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_dim = 768  # Adjust if not BERT-base
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.emb_dim * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_labels)
        )
        self.batch_size = batch_size
        self.device = device

    def forward(self, sentence1, sentence2):
        all_logits = []
        for i in range(0, len(sentence1), self.batch_size):
            batch_1 = sentence1[i:i + self.batch_size]
            batch_2 = sentence2[i:i + self.batch_size]

            emb_1 = self.encode(batch_1, convert_to_tensor=True).to(self.device)
            emb_2 = self.encode(batch_2, convert_to_tensor=True).to(self.device)

            combined = torch.cat([emb_1, emb_2], dim=1)
            logits = self.classifier(combined)
            all_logits.append(logits)

        return torch.cat(all_logits, dim=0)


def load_sts_model(checkpoint_path, device='cpu', num_labels=2):
    args = type('Args', (), {})()
    args.padding_side = 'right'
    args.num_heads = 8
    args.ln = True
    args.norm = True
    args.neg_K = 4

    model = STSClassifier(
        model_name_or_path='bert-base-uncased',
        max_seq_length=256,
        args=args,
        batch_size=32,
        device=device,
        num_labels=num_labels
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def run_sts_inference(model, sts_test_path, device, label_column='score'):
    df = pd.read_csv(sts_test_path, sep=',')

    sentence1 = df['sentence1'].tolist()
    sentence2 = df['sentence2'].tolist()

    # For binary classification (similar vs not similar)
    # If your label is a score (e.g., 0–5), binarize it here (adjust threshold as needed)
    df[label_column] = df[label_column].apply(lambda x: 1 if float(x) > 2.5 else 0)
    true_labels = df[label_column].tolist()

    with torch.no_grad():
        logits = model(sentence1, sentence2)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().tolist()

    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, average='macro', zero_division=0)
    rec = recall_score(true_labels, preds, average='macro', zero_division=0)
    f1 = f1_score(true_labels, preds, average='macro', zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = "../PATH_TO_OUTPUT_MODEL/sts/final_student_model_fp32/pytorch_model.bin"
    sts_test_path = "../dataset/sts/test.csv"  # Must have columns: sentence1, sentence2, score

    model = load_sts_model(checkpoint_path, device, num_labels=2)  # Use num_labels=6 for STS-B (0–5)
    run_sts_inference(model, sts_test_path, device)
