# train_analyzer.py
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

# 1. Load the processed dataset and label maps
print("🔹 Loading dataset and label maps...")
dataset = load_dataset('json', data_files='processed_data_fixed.jsonl', split='train')

with open("label_maps_fixed.json", "r") as f:
    label_maps = json.load(f)

id2label = {int(k): v for k, v in label_maps['id2label'].items()}
label2id = label_maps['label2id']

num_labels = len(id2label)
print(f"✅ Loaded {num_labels} labels.")

# 2. Preprocess the data
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

train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# 3. Load the Longformer Model
print(f"🔹 Loading {MODEL_NAME} model for multi-label classification...")
model = LongformerForSequenceClassification.from_pretrained(
    MODEL_NAME,
    problem_type="multi_label_classification",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

# 4. Define Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,              # Increase later for better results
    per_device_train_batch_size=4,   # safe for Colab T4
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",           # ✅ updated keyword
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,                       # mixed precision = faster training
)

# 5. Define Evaluation Metric
def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    predictions = (probs > 0.5).int().numpy()

    f1 = f1_score(y_true=p.label_ids, y_pred=predictions, average='samples', zero_division=0)
    return {'f1_samples': f1}

# 6. Create and run the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

print("🚀 Starting model training on Colab T4...")
trainer.train()

# 7. Save the final model
model_save_path = "./fine-tuned-analyzer"
print(f"💾 Saving the fine-tuned model to {model_save_path}...")
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("✅ Model training complete!")
