import os
import json
import torch
import re
from tqdm import tqdm
import pytest
from argparse import Namespace

from model.pro_model import Mymodel

MODEL_PATH = "../PATH_TO_OUTPUT_MODEL/webq/final_student_model_fp32"
TEST_FILE = "../dataset/web_questions/web_questions_test.json"

args = Namespace(
    num_heads=8,
    ln=True,
    norm=True,
    padding_side='right',
    neg_K=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_answer(s):
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return re.sub(r'[^\w\s]', '', text)
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_em_f1(pred, golds):
    pred_norm = normalize_answer(pred)
    em = max(pred_norm == normalize_answer(g) for g in golds)
    f1s = []
    for g in golds:
        g_norm = normalize_answer(g)
        pred_tokens = pred_norm.split()
        gold_tokens = g_norm.split()
        common = set(pred_tokens) & set(gold_tokens)
        if not common:
            f1s.append(0)
        else:
            precision = len(common) / len(pred_tokens)
            recall = len(common) / len(gold_tokens)
            f1s.append(2 * precision * recall / (precision + recall))
    return em, max(f1s)


@pytest.mark.eval
def test_run_webq_eval():
    assert os.path.exists(TEST_FILE), f"Test file not found: {TEST_FILE}"
    assert os.path.exists(MODEL_PATH), f"Model path not found: {MODEL_PATH}"

    model = Mymodel(model_name_or_path=MODEL_PATH, args=args)
    model.eval()
    model.to(device)

    examples = []
    with open(TEST_FILE, "r") as f:
        for line in tqdm(f, desc="Loading test samples", total=2000):  # adjust total if needed
            examples.append(json.loads(line.strip()))

    em_total, f1_total = 0, 0
    results = []
    imperfect_results = []

    for idx, ex in enumerate(examples):
        question = ex["question"]
        gold_answers = ex["answers"]
        candidate_answers = ex.get("candidate_answers", gold_answers)

        with torch.no_grad():
            question_emb = model.encode(question, convert_to_tensor=True).to(device)
            option_embs = model.encode(candidate_answers, convert_to_tensor=True).to(device)

            question_expanded = question_emb.unsqueeze(0).expand(option_embs.size(0), -1)
            task_ids = torch.zeros(option_embs.size(0), dtype=torch.long).to(device)

            logits, _ = model.iem(question_expanded, option_embs)
            scores = logits[:, 0]
            best_index = torch.argmax(scores).item()
            best_answer = candidate_answers[best_index]

        em, f1 = compute_em_f1(best_answer, gold_answers)
        em_total += em
        f1_total += f1

        result = {
            "question": question,
            "predicted_answer": best_answer,
            "gold_answers": gold_answers,
            "em": em,
            "f1": f1
        }
        results.append(result)

        # Print first 10 examples for inspection
        if idx < 10:
            print("\n--- Example ---")
            print(f"Question        : {question}")
            print(f"Predicted Answer: {best_answer}")
            print(f"Gold Answers    : {gold_answers}")
            print(f"EM              : {em}")
            print(f"F1              : {f1:.4f}")

        # Log imperfect predictions
        if em != 1 or f1 < 1:
            print("\n>>> [Warning] Imperfect Prediction Detected")
            print(f"Question        : {question}")
            print(f"Predicted Answer: {best_answer}")
            print(f"Gold Answers    : {gold_answers}")
            print(f"EM              : {em}")
            print(f"F1              : {f1:.4f}")
            print("---")

            imperfect_results.append({
                "question": question,
                "predicted_answer": best_answer,
                "gold_answers": gold_answers,
                "normalized_pred": normalize_answer(best_answer),
                "normalized_gold": [normalize_answer(g) for g in gold_answers],
                "em": em,
                "f1": f1
            })

    avg_em = em_total / len(results)
    avg_f1 = f1_total / len(results)

    print(f"\nEval finished on {len(results)} examples")
    print(f"Avg EM: {avg_em:.2%}")
    print(f"Avg F1: {avg_f1:.2%}")

    if imperfect_results:
        with open("imperfect_predictions.json", "w") as f:
            json.dump(imperfect_results, f, indent=2)

    # Optional assertion to catch issues in automated tests
    assert avg_f1 > 0.1, "F1 score too low — possible model or data issue"
