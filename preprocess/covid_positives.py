import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, \
    TrainingArguments, Trainer
import csv
import pickle
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import random
import os


def covid_template_context(query, passage):
    return (
        f"#Q describes a user question. #A describes a web passage. "
        f"Are they related such that #A correctly answers #Q? Answer Can or Cannot.\n"
        f"#Q: {query}\n#A: {passage}\nAnswer:"
    )


def get_simple_embeddings(outputs, attention_mask):
    """Simple mean pooling without enhancement"""
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask

    return embeddings


class CovidDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(p):
    """Compute metrics for fine-tuning"""
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='binary')
    acc = accuracy_score(p.label_ids, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def create_finetuning_data_optimized(positive_tsv_path, num_positive_samples=5000, num_negative_samples=5000):
    """Create smaller balanced training data for faster fine-tuning"""

    # Load positive examples - take a subset
    positive_pairs = []
    with open(positive_tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for i, row in enumerate(reader):
            if i >= num_positive_samples:  # Limit positive examples
                break
            question = row['sentence1'].strip()
            answer = row['sentence2'].strip()
            if question and answer:
                template = covid_template_context(question, answer)
                positive_pairs.append((template, 0))  # 0 = "Can" answer

    print(f"Loaded {len(positive_pairs)} positive examples (subset)")

    # Extract actual questions and answers from templates
    actual_questions = []
    actual_answers = []

    for template, _ in positive_pairs:
        try:
            q_start = template.find("#Q: ") + 4
            q_end = template.find("\n#A:")
            a_start = template.find("#A: ") + 4
            a_end = template.find("\nAnswer:")

            actual_q = template[q_start:q_end].strip()
            actual_a = template[a_start:a_end].strip()
            actual_questions.append(actual_q)
            actual_answers.append(actual_a)
        except:
            continue

    # Create unique lists
    unique_questions = list(set(actual_questions))
    unique_answers = list(set(actual_answers))

    print(f"Unique questions: {len(unique_questions)}")
    print(f"Unique answers: {len(unique_answers)}")

    # Generate negative examples
    negative_pairs = []
    max_attempts = num_negative_samples * 3
    attempts = 0

    while len(negative_pairs) < num_negative_samples and attempts < max_attempts:
        q = random.choice(unique_questions)
        a = random.choice(unique_answers)

        # Create template and check if it's a positive pair
        template = covid_template_context(q, a)
        is_positive = any(template == pos_template for pos_template, _ in positive_pairs)

        if not is_positive:
            negative_pairs.append((template, 1))  # 1 = "Cannot" answer

        attempts += 1

    print(f"Created {len(negative_pairs)} negative examples")

    # Combine and shuffle
    all_data = positive_pairs + negative_pairs
    random.shuffle(all_data)

    all_texts = [text for text, _ in all_data]
    all_labels = [label for _, label in all_data]

    print(f"Total training data: {len(all_texts)} (Pos: {len(positive_pairs)}, Neg: {len(negative_pairs)})")

    return all_texts, all_labels


def fine_tune_distilbert_classifier(train_texts, train_labels, val_texts=None, val_labels=None,
                                    output_dir='./distilbert-covid-classifier'):
    """Fine-tune DistilBERT for faster training"""

    print("Loading DistilBERT tokenizer and model for faster fine-tuning...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    ).cuda()

    # Tokenize training data
    print("Tokenizing training data...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=256
    )

    train_dataset = CovidDataset(train_encodings, train_labels)

    # Tokenize validation data if provided
    if val_texts is not None and val_labels is not None:
        print("Tokenizing validation data...")
        val_encodings = tokenizer(
            val_texts,
            truncation=True,
            padding=True,
            max_length=256
        )
        val_dataset = CovidDataset(val_encodings, val_labels)
    else:
        val_dataset = None
        print("No validation data provided")

    # Optimized training arguments for DistilBERT
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,  # Reduced epochs
        per_device_train_batch_size=32,  # Larger batch size (DistilBERT is smaller)
        per_device_eval_batch_size=32,
        warmup_steps=100,  # Reduced warmup
        learning_rate=5e-5,  # Slightly higher learning rate
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=200,  # More frequent evaluation
        save_steps=400,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        report_to=None,
        fp16=True,  # Mixed precision for faster training
        dataloader_pin_memory=False,
        gradient_accumulation_steps=1,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("Starting DistilBERT fine-tuning...")
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(f'{output_dir}-final')
    tokenizer.save_pretrained(f'{output_dir}-final')

    print(f"Fine-tuned DistilBERT model saved to {output_dir}-final")

    return model, tokenizer


def get_finetuned_distilbert_classification_logits(texts, model_path, batch_size=32, max_length=256):
    """Use fine-tuned DistilBERT for classification scores"""
    print(f"Loading fine-tuned DistilBERT from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
    model.eval()

    all_logits = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating DistilBERT classification logits"):
            batch_texts = texts[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to('cuda')

            outputs = model(**inputs)
            logits = outputs.logits

            # Convert to probabilities
            probs = torch.softmax(logits.float(), dim=-1)
            all_logits.extend(probs.cpu().numpy().tolist())

    return all_logits


def generate_positive_logits_and_features_distilbert(tsv_path, output_path, finetuned_model_path, batch_size=32,
                                                     max_length=256):
    """DISTILBERT VERSION: Use fine-tuned DistilBERT classifier for faster performance"""

    # Load data
    pairs = []
    question_count = defaultdict(int)

    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            question = row['sentence1'].strip()
            answer = row['sentence2'].strip()
            if question and answer:
                pairs.append((question, answer))
                question_count[question] += 1

    print(f"Loaded {len(pairs)} positive pairs")
    print(f"Unique questions: {len(question_count)}")

    # Show duplicate statistics
    duplicates = {q: count for q, count in question_count.items() if count > 1}
    if duplicates:
        print(f"Questions with multiple answers: {len(duplicates)}")
        for q, count in list(duplicates.items())[:3]:
            print(f"  '{q}' -> {count} answers")

    # Use BERT for feature extraction
    print("Loading sentence-transformers model for feature extraction...")
    embed_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    embed_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').cuda()
    embed_model.eval()

    # 🔥 DISTILBERT: Use fine-tuned DistilBERT classifier
    print("Using fine-tuned DistilBERT classifier for faster logit generation...")
    all_templates = [covid_template_context(q, a) for q, a in pairs]
    all_logits = get_finetuned_distilbert_classification_logits(
        all_templates,
        finetuned_model_path,
        batch_size=batch_size,
        max_length=max_length
    )

    # Generate features from answers
    print("Generating features from answers...")
    all_features = []
    answers_only = [a for q, a in pairs]

    with torch.no_grad():
        for i in tqdm(range(0, len(answers_only), batch_size), desc="Answer features"):
            batch_answers = answers_only[i:i + batch_size]
            inputs = embed_tokenizer(batch_answers, padding=True, truncation=True,
                                     max_length=max_length, return_tensors="pt").to('cuda')
            outputs = embed_model(**inputs)

            # Use simple embedding generation (no enhancement)
            embeddings = get_simple_embeddings(outputs, inputs['attention_mask'])
            features = F.normalize(embeddings, p=2, dim=1)
            all_features.append(features.cpu().to(torch.float16))

    # 🔥 ENHANCED: Store logits keyed by (question, answer) pairs with quality metrics
    output_dict = {
        'logits': {},
        'features': {},
        'metadata': {
            'total_pairs': len(pairs),
            'unique_questions': len(question_count),
            'questions_with_multiple_answers': len(duplicates),
            'feature_similarity': 0.0,
            'logit_quality': {},
            'key_format': 'question_answer_pair',
            'model_used': 'fine-tuned-distilbert'
        }
    }

    all_features_combined = torch.cat(all_features, dim=0) if all_features else torch.tensor([])

    # Store logits and features keyed by (question, answer) pairs
    for idx, ((question, answer), logit) in enumerate(zip(pairs, all_logits)):
        # Key by the actual pair for exact matching
        pair_key = (question, answer)
        output_dict['logits'][pair_key] = logit

        if idx < len(all_features_combined):
            output_dict['features'][pair_key] = all_features_combined[idx].numpy().tolist()

    # Calculate feature similarity
    if len(all_features_combined) > 0:
        features_tensor = all_features_combined.float()
        similarity_matrix = torch.mm(features_tensor, features_tensor.T)
        mask = ~torch.eye(similarity_matrix.size(0)).bool()
        avg_similarity = similarity_matrix[mask].mean().item()
        output_dict['metadata']['feature_similarity'] = avg_similarity

    # 🔥 ENHANCED: Calculate logit quality metrics
    all_logits_tensor = torch.tensor(all_logits)
    can_probs = all_logits_tensor[:, 0]  # "Can" probabilities
    cannot_probs = all_logits_tensor[:, 1]  # "Cannot" probabilities

    avg_can = can_probs.mean().item()
    avg_cannot = cannot_probs.mean().item()
    separation = avg_can - avg_cannot
    discrimination_accuracy = (can_probs > cannot_probs).float().mean().item()

    output_dict['metadata']['logit_quality'] = {
        'avg_can_prob': avg_can,
        'avg_cannot_prob': avg_cannot,
        'separation': separation,
        'discrimination_accuracy': discrimination_accuracy,
        'strong_discrimination': ((can_probs - cannot_probs) > 0.3).float().mean().item(),
        'weak_discrimination': ((can_probs - cannot_probs) < 0.1).float().mean().item(),
        'confidence_mean': (can_probs - cannot_probs).abs().mean().item(),
        'confidence_std': (can_probs - cannot_probs).abs().std().item()
    }

    # Save
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)

    # 🔥 ENHANCED QUALITY REPORTING
    print(f"\n🎯 DISTILBERT POSITIVE LOGIT QUALITY ANALYSIS:")
    print(f"   Average 'Can' probability: {avg_can:.4f}")
    print(f"   Average 'Cannot' probability: {avg_cannot:.4f}")
    print(f"   Separation (Can - Cannot): {separation:.4f}")
    print(f"   Discrimination accuracy: {discrimination_accuracy * 100:.1f}%")
    print(
        f"   Strong discrimination (>0.3): {output_dict['metadata']['logit_quality']['strong_discrimination'] * 100:.1f}%")
    print(
        f"   Weak discrimination (<0.1): {output_dict['metadata']['logit_quality']['weak_discrimination'] * 100:.1f}%")
    print(f"   Average confidence: {output_dict['metadata']['logit_quality']['confidence_mean']:.4f}")

    if separation > 0.6:
        print("   🎉 EXCELLENT: Outstanding positive discrimination!")
    elif separation > 0.4:
        print("   🎉 EXCELLENT: Strong positive discrimination!")
    elif separation > 0.2:
        print("   ✅ GOOD: Clear positive discrimination")
    elif separation > 0.1:
        print("   ⚠️  ACCEPTABLE: Minimal positive discrimination")
    else:
        print("   🚨 POOR: Barely any discrimination")

    print(f"✅ Feature similarity: {avg_similarity:.4f}")
    if avg_similarity < 0.3:
        print("🎉 Excellent feature diversity!")
    elif avg_similarity < 0.5:
        print("✅ Good feature diversity")
    else:
        print("⚠️  Feature diversity could be improved")

    print(f"\n✅ DISTILBERT VERSION: Saved to {output_path}")
    print(f"   - Total pairs: {len(pairs)}")
    print(f"   - Logit entries: {len(output_dict['logits'])}")
    print(f"   - Feature entries: {len(output_dict['features'])}")
    print(f"   - Feature similarity: {output_dict['metadata']['feature_similarity']:.4f}")
    print(f"   - Model: {output_dict['metadata']['model_used']}")

    # Show samples
    print(f"\n📋 Sample logits (first 3):")
    for i, (pair_key, logit) in enumerate(list(output_dict['logits'].items())[:3]):
        question, answer = pair_key
        can_prob, cannot_prob = logit
        confidence = abs(can_prob - cannot_prob)
        print(f"  {i + 1}. Q: '{question[:50]}...'")
        print(f"     A: '{answer[:50]}...'")
        print(f"     Can: {can_prob:.4f}, Cannot: {cannot_prob:.4f}, Confidence: {confidence:.4f}")

    return output_dict


def create_simple_positive_logits_final():
    """Create training-compatible version with proper (question, answer) keys"""

    for split in ['train', 'val']:
        input_path = f'../outputs/pos_emb/covid_{split}_pos_emb_distilbert.pkl'
        output_path = f'../outputs/pos_emb/covid_{split}_pos_logits_distilbert.pkl'

        print(f"\n🔄 Creating training logits for {split}...")

        with open(input_path, 'rb') as f:
            data = pickle.load(f)

        # For training: extract logits and create flat dictionary with (question, answer) keys
        training_logits = {}

        # Check if we have the new format with pair keys
        if data['logits'] and isinstance(next(iter(data['logits'].keys())), tuple):
            # Already in (question, answer) format - just extract the logits
            for pair_key, logit in data['logits'].items():
                training_logits[pair_key] = logit
        else:
            # Old format - need to reconstruct pairs (this shouldn't happen with the new code)
            print(f"⚠️  Old format detected for {split}, reconstructing pairs...")
            # You would need the original TSV data to reconstruct pairs properly

        with open(output_path, 'wb') as f:
            pickle.dump(training_logits, f)

        # Calculate training logit quality
        all_training_logits = list(training_logits.values())
        if all_training_logits:
            logits_tensor = torch.tensor(all_training_logits)
            can_probs = logits_tensor[:, 0]
            cannot_probs = logits_tensor[:, 1]
            separation = (can_probs - cannot_probs).mean().item()
            accuracy = (can_probs > cannot_probs).float().mean().item()

            print(f"📊 Training Logit Quality for {split}:")
            print(f"   • Separation: {separation:.4f}")
            print(f"   • Accuracy: {accuracy * 100:.1f}%")
            print(f"   • Total pairs: {len(training_logits)}")

        print(f"✅ Saved {len(training_logits)} training logits to {output_path}")

        # Show samples
        print(f"📋 Sample training logits for {split}:")
        for i, (pair_key, logit) in enumerate(list(training_logits.items())[:2]):
            question, answer = pair_key
            can_prob, cannot_prob = logit
            print(f"  {i + 1}. Q: '{question[:30]}...' -> A: '{answer[:30]}...'")
            print(f"     Logit: Can={can_prob:.4f}, Cannot={cannot_prob:.4f}")


def main():
    print("=== OPTIMIZED VERSION: DISTILBERT WITH SMALLER DATASET FOR FASTER TRAINING ===")

    # Step 1: Create smaller training data and fine-tune DistilBERT
    print("🎯 Step 1: Creating smaller training data and fine-tuning DistilBERT...")

    # Create smaller training data from train positives
    train_texts, train_labels = create_finetuning_data_optimized(
        "../dataset/covid/train/positives.tsv",
        num_positive_samples=5000,    # Reduced from 38K to 5K
        num_negative_samples=5000     # Reduced from 20K to 5K
    )

    # Create smaller validation data from val positives
    val_texts, val_labels = create_finetuning_data_optimized(
        "../dataset/covid/val/positives.tsv",
        num_positive_samples=1000,    # Reduced from 4K to 1K
        num_negative_samples=1000     # Reduced from 2K to 1K
    )

    # Fine-tune DistilBERT (much faster than BERT)
    finetuned_model, finetuned_tokenizer = fine_tune_distilbert_classifier(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        output_dir='./distilbert-covid-classifier'
    )

    finetuned_model_path = './distilbert-covid-classifier-final'

    print(f"\n🎯 Step 2: Generating logits with fine-tuned DistilBERT model...")

    # Generate logits with fine-tuned DistilBERT model
    train_output = generate_positive_logits_and_features_distilbert(
        tsv_path="../dataset/covid/train/positives.tsv",
        output_path="../outputs/pos_emb/covid_train_pos_emb_distilbert.pkl",
        finetuned_model_path=finetuned_model_path
    )

    print("\n=== VALIDATION GENERATION ===")
    val_output = generate_positive_logits_and_features_distilbert(
        tsv_path="../dataset/covid/val/positives.tsv",
        output_path="../outputs/pos_emb/covid_val_pos_emb_distilbert.pkl",
        finetuned_model_path=finetuned_model_path
    )

    # Create training versions
    create_simple_positive_logits_final()

    # 🔥 FINAL COMPARISON SUMMARY
    print("\n" + "=" * 60)
    print("🎯 DISTILBERT POSITIVE LOGIT GENERATION - QUALITY SUMMARY")
    print("=" * 60)

    for split, output in [('TRAIN', train_output), ('VAL', val_output)]:
        quality = output['metadata']['logit_quality']
        print(f"\n{split}:")
        print(f"   • Separation: {quality['separation']:.4f}")
        print(f"   • Accuracy: {quality['discrimination_accuracy'] * 100:.1f}%")
        print(f"   • Strong discrimination: {quality['strong_discrimination'] * 100:.1f}%")
        print(f"   • Average confidence: {quality['confidence_mean']:.4f}")
        print(f"   • Feature similarity: {output['metadata']['feature_similarity']:.4f}")

        if quality['separation'] > 0.6:
            print("   🎉 OUTSTANDING quality!")
        elif quality['separation'] > 0.4:
            print("   🎉 EXCELLENT quality!")
        elif quality['separation'] > 0.2:
            print("   ✅ GOOD quality")
        elif quality['separation'] > 0.1:
            print("   ⚠️  ACCEPTABLE quality")
        else:
            print("   🚨 POOR quality")

    print(f"\n🎉 DISTILBERT GENERATION COMPLETE ===")
    print(f"Training: {train_output['metadata']['total_pairs']} pairs")
    print(f"Validation: {val_output['metadata']['total_pairs']} pairs")
    print(f"Fine-tuned model: {finetuned_model_path}")
    print("✅ Ready for training with fast fine-tuned DistilBERT classifier logits!")


if __name__ == "__main__":
    main()