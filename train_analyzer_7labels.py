# train_analyzer_7labels.py
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score

# -------------------------------
# 1. Load dataset + label maps
# -------------------------------
print("🔹 Loading dataset...")
dataset = load_dataset('json', data_files='processed_data_7labels.jsonl', split='train')

TARGET_LABELS = ['joy', 'anger', 'sadness', 'fear', 'love', 'surprise', 'neutral']
id2label = {i: l for i, l in enumerate(TARGET_LABELS)}
label2id = {l: i for i, l in id2label.items()}
num_labels = len(TARGET_LABELS)

print(f"✅ Labels: {id2label}")

# -------------------------------
# 2. Tokenizer
# -------------------------------
MODEL_NAME = 'allenai/longformer-base-4096'
print(f"🔹 Loading tokenizer for {MODEL_NAME}...")
tokenizer = LongformerTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=512
    )

print("🔹 Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)

train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# -------------------------------
# 3. Compute Class Weights
# -------------------------------
print("🔹 Computing class weights...")
label_matrix = np.array([ex["labels"] for ex in dataset])
label_counts = label_matrix.sum(axis=0)
print("Label counts:", dict(zip(TARGET_LABELS, label_counts)))

# Avoid div by zero → add small epsilon
total = len(dataset)
pos_weights = (total - label_counts) / (label_counts + 1e-5)
class_weights = torch.tensor(pos_weights, dtype=torch.float)
print("Class weights:", class_weights)

# -------------------------------
# 4. Load Longformer
# -------------------------------
print(f"🔹 Loading {MODEL_NAME}...")
model = LongformerForSequenceClassification.from_pretrained(
    MODEL_NAME,
    problem_type="multi_label_classification",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# Replace default loss with weighted BCE
def custom_loss(model, inputs, return_outputs=False):
    labels = inputs.get("labels")
    outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
    logits = outputs.logits
    loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=class_weights.to(logits.device))
    loss = loss_fct(logits, labels.float())
    return (loss, outputs) if return_outputs else loss

# -------------------------------
# 5. Training Args
# -------------------------------
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # train longer
    per_device_train_batch_size=4,   # safe for Colab T4
    per_device_eval_batch_size=4,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    save_total_limit=2,
)

# -------------------------------
# 6. Metrics
# -------------------------------
def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    preds = (probs > 0.5).int().numpy()
    f1_micro = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    f1_macro = f1_score(y_true=p.label_ids, y_pred=preds, average='macro', zero_division=0)
    return {"f1_micro": f1_micro, "f1_macro": f1_macro}

# -------------------------------
# 7. Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    loss_func=custom_loss,  # ✅ use weighted loss
)

print("🚀 Starting training...")
trainer.train()

# -------------------------------
# 8. Save final model
# -------------------------------
save_path = "/content/drive/MyDrive/fine-tuned-analyzer-7labels"
print(f"💾 Saving model to {save_path}...")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)
print("✅ Training complete!")
