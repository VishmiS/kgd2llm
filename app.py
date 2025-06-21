import os
import logging
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer
from models import LoRAStudentModel
import re
import nltk
from nltk.corpus import stopwords
from lime.lime_text import LimeTextExplainer
import numpy as np
import string

# Download stopwords from nltk if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model and tokenizer
model_path = './student_model/final_student_model.pt'
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Use BERT tokenizer

# Set padding token and verify
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    logger.info(f"Padding token set to: {tokenizer.pad_token}")
    logger.info(f"Teacher vocab size: {tokenizer.vocab_size}")

# Load model and set it to evaluation mode
model = LoRAStudentModel(model_name='bert-base-uncased', num_classes=3)
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
        logger.info("Model loaded successfully with strict=False.")
    except RuntimeError as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

model.eval()

# Label mapping for model output
label_mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}

# Define a function for LIME to get probabilities from the model
def predict_proba(texts):
    model.eval()
    probs_list = []

    for text in texts:
        encodings = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.nn.functional.softmax(logits, dim=1)
            probs_list.append(probs.cpu().numpy())

    return np.concatenate(probs_list, axis=0)

@app.route('/sts/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        sentence1 = data.get("sentence1", "")
        sentence2 = data.get("sentence2", "")

        if not sentence1 or not sentence2:
            return jsonify({"error": "Both sentences are required."}), 400

        combined_input = f"{sentence1} {sentence2}"
        inputs = tokenizer(combined_input, return_tensors="pt", truncation=True, padding=True, max_length=256)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']

        with torch.no_grad():
            model_output = model(input_ids, attention_mask=attention_mask)
            logits = model_output[0] if isinstance(model_output, tuple) else model_output
            probabilities = torch.nn.functional.softmax(logits, dim=1)

            predicted_label = torch.argmax(probabilities).item()
            confidence_score = probabilities[0, predicted_label].item() * 100

            probabilities_dict = {
                label_mapping[i]: probabilities[0, i].item() for i in range(probabilities.size(1))
            }

        class_names = ["entailment", "neutral", "contradiction"]
        explainer = LimeTextExplainer(class_names=class_names)

        explanation = explainer.explain_instance(
            f"{sentence1} {sentence2}",
            predict_proba,
            num_features=10,
            num_samples=200  # or 100–300 for faster performance, removing this line will provide faster performance.
        )
        lime_explanation = explanation.as_list()

        highlighted_data = highlight_keywords(sentence1, sentence2, predicted_label, lime_explanation)

        response = {
            "predicted_label": label_mapping[predicted_label],
            "confidence_score": confidence_score,
            "all_class_probabilities": probabilities_dict,
            "highlighted_keywords": highlighted_data,
            "lime_explanation": {
                "keywords": [word for word, _ in lime_explanation],
                "weights": [weight for _, weight in lime_explanation]
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_color_from_weight(weight):
    """
    Simplified color mapping based on LIME weight:
    - Positive weights -> Green
    - Negative weights -> Red
    - Near Zero weights -> Gray
    """
    if weight > 0:
        # Positive weight -> Green
        color = "rgba(0, 255, 0, 1.00)"  # Solid Green
    elif weight < 0:
        # Negative weight -> Red
        color = "rgba(255, 0, 0, 1.00)"  # Solid Red
    else:
        # Near zero -> Gray
        color = "rgba(128, 128, 128, 1.00)"  # Solid Gray

    return color

def highlight_keywords(sentence1, sentence2, predicted_label, lime_explanation):
    """
    Highlights keywords from LIME explanation in sentence1 and sentence2
    using color coding based on weight.
    """
    # Build a dictionary of words and their weights from the explanation
    lime_dict = {word.lower(): weight for word, weight in lime_explanation}

    def highlight_text(sentence):
        # Tokenize keeping punctuation for display
        words = sentence.split()
        highlighted_sentence = []

        for word in words:
            clean_word = word.strip(string.punctuation)
            weight = lime_dict.get(clean_word.lower(), None)

            if weight is not None:
                color = get_color_from_weight(weight)
                marked_word = f'<mark style="background-color: {color};" data-weight="{weight}" class="highlight">{word}</mark>'
                highlighted_sentence.append(marked_word)
            else:
                highlighted_sentence.append(word)

        return " ".join(highlighted_sentence)

    return {
        "keywords": list(lime_dict.keys()),
        "sentence1": highlight_text(sentence1),
        "sentence2": highlight_text(sentence2)
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
