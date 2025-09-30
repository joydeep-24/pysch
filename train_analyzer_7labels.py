# train_analyzer_7labels.py
import torch
import json
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
print("🔹 Loading dataset and label maps...")
dataset = load_dataset('json', data_files='processed_data_7labels.jsonl', split='train')

with open("label_maps_7labels.json", "r") as f:
    label_maps = json.load(f)

id2label = {int(k): v for k, v in label_maps['id2label'].items()}
label2id = label_maps['label2id']

num_labels = len(id2label)
print(f"✅ Loaded {num_labels} labels: {list(id2label.values())}")

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
# 3. Load Longformer
# -------------------------------
print(f"🔹 Loading {MODEL_NAME} model for multi-label classification...")
model = LongformerForSequenceClassification.from_pretrained(
    MODEL_NAME,
    problem_type="multi_label_classification",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# -------------------------------
# 4. Training Args
# -------------------------------
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,               # bump up for better performance
    per_device_train_batch_size=4,    # fits on Colab T4
    per_device_eval_batch_size=4,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",      # ✅ correct arg
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                        # mixed precision on T4
    save_total_limit=2
)

# -------------------------------
# 5. Metrics
# -------------------------------
def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    preds = (probs > 0.5).int().numpy()

    f1 = f1_score(y_true=p.label_ids, y_pred=preds, average='micro', zero_division=0)
    return {'f1_micro': f1}

# -------------------------------
# 6. Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print("🚀 Starting model training...")
trainer.train()

# -------------------------------
# 7. Save final model
# -------------------------------
model_save_path = "./fine-tuned-analyzer-7labels"
print(f"💾 Saving the fine-tuned model to {model_save_path}...")
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("✅ Training complete!")
