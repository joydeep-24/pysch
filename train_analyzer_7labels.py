# train_analyzer_7labels.py
import torch
import json
import numpy as np
from datasets import load_dataset
from transformers import (
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score
import torch.nn as nn

# -------------------------------
# 1. Load dataset
# -------------------------------
print("🔹 Loading dataset...")
dataset = load_dataset('json', data_files='processed_data_7labels.jsonl', split='train')

# Define label maps
id2label = {0: "joy", 1: "anger", 2: "sadness", 3: "fear", 4: "love", 5: "surprise", 6: "neutral"}
label2id = {v: k for k, v in id2label.items()}

num_labels = len(id2label)
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

# Train-test split
train_test_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# -------------------------------
# 3. Compute Class Weights
# -------------------------------
print("🔹 Computing class weights...")
all_labels = torch.tensor([ex["labels"] for ex in dataset])
label_counts = torch.sum(all_labels, dim=0).numpy()

total_samples = len(dataset)
class_weights = total_samples / (len(id2label) * label_counts)
class_weights = torch.tensor(class_weights, dtype=torch.float)

print("Label counts:", {id2label[i]: label_counts[i] for i in range(num_labels)})
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
    label2id=label2id
)

# -------------------------------
# 5. Training Args
# -------------------------------
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,              # increase for better results
    per_device_train_batch_size=4,   # Colab T4 safe
    per_device_eval_batch_size=4,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",     # ✅ correct argument name
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    save_total_limit=2
)

# -------------------------------
# 6. Metrics
# -------------------------------
def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    preds = (probs > 0.5).int().numpy()

    f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    return {'f1_micro': f1}

# -------------------------------
# 7. Weighted Trainer
# -------------------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device) if class_weights is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights) if self.class_weights is not None else nn.BCEWithLogitsLoss()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # ✅ accept extra kwargs
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fn(logits, labels.float())
        return (loss, outputs) if return_outputs else loss

# -------------------------------
# 8. Trainer
# -------------------------------
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    class_weights=class_weights
)

print("🚀 Starting model training...")
trainer.train()

# -------------------------------
# 9. Save final model
# -------------------------------
model_save_path = "/content/fine-tuned-analyzer-7labels"
print(f"💾 Saving the fine-tuned model to {model_save_path}...")
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("✅ Training complete!")
