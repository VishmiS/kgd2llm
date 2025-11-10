import pickle

# Test the training logits file
print("=== TRAINING LOGITS FILE ===")
with open('webq_train_pos_emb.pkl', 'rb') as f:
    train_logits = pickle.load(f)
print(f"Entries: {len(train_logits)}")
sample_key = next(iter(train_logits.keys()))
print(f"Sample key: {type(sample_key)} - {sample_key}")
print(f"Sample value: {train_logits[sample_key]}")

print("\n=== VALIDATION LOGITS FILE ===")
with open('webq_val_pos_emb.pkl', 'rb') as f:
    val_logits = pickle.load(f)
print(f"Entries: {len(val_logits)}")