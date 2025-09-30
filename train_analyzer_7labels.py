import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import (
    LongformerTokenizerFast,
    LongformerForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Load dataset
# -------------------------------
print("🔹 Loading dataset...")
dataset = load_dataset("json", data_files="processed_data_7labels.jsonl", split="train")

# Define labels
id2label = {0: "joy", 1: "anger", 2: "sadness", 3: "fear", 4: "love", 5: "surprise", 6: "neutral"}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)
print(f"✅ Labels: {id2label}")

# -------------------------------
# 2. Tokenizer with optimizations
# -------------------------------
MODEL_NAME = "allenai/longformer-base-4096"
print(f"🔹 Loading tokenizer for {MODEL_NAME}...")
tokenizer = LongformerTokenizerFast.from_pretrained(MODEL_NAME)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    # Use dynamic padding instead of max_length for efficiency
    return tokenizer(
        examples["text"],
        padding="max_length",  # Can change to "longest" for dynamic padding
        truncation=True,
        max_length=256  # Reduced from 512 - most texts are shorter
    )

print("🔹 Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Better train-validation-test split
splits = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
train_val = splits["train"]
test_dataset = splits["test"]

# Further split train into train and validation
train_val_splits = train_val.train_test_split(test_size=0.125, seed=42)  # 0.125 of 0.8 = 0.1 total
train_dataset = train_val_splits["train"]
eval_dataset = train_val_splits["test"]

print(f"📊 Dataset sizes - Train: {len(train_dataset)}, Val: {len(eval_dataset)}, Test: {len(test_dataset)}")

# -------------------------------
# 3. Improved Class Weights Calculation
# -------------------------------
print("🔹 Computing improved class weights...")
all_labels = torch.tensor(dataset["labels"])
label_counts = all_labels.sum(dim=0).numpy()

# Calculate positive and negative samples for each class
total_samples = len(dataset)
neg_counts = total_samples - label_counts

# Compute balanced weights (handles class imbalance better)
pos_weights = neg_counts / (label_counts + 1e-5)  # Avoid division by zero
pos_weights = torch.tensor(pos_weights, dtype=torch.float)

# Clip weights to prevent extreme values
pos_weights = torch.clamp(pos_weights, min=0.5, max=10.0)

print("Label distribution:")
for i in range(num_labels):
    print(f"  {id2label[i]}: {int(label_counts[i])} samples, weight: {pos_weights[i]:.2f}")

# -------------------------------
# 4. Load Model with better initialization
# -------------------------------
print(f"🔹 Loading {MODEL_NAME}...")
model = LongformerForSequenceClassification.from_pretrained(
    MODEL_NAME,
    problem_type="multi_label_classification",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    hidden_dropout_prob=0.2,  # Add dropout for regularization
    attention_probs_dropout_prob=0.2
)

# Initialize classifier layer with Xavier initialization
def init_weights(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()

model.classifier.apply(init_weights)

# -------------------------------
# 5. Optimized Training Arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir="./results_improved",
    num_train_epochs=3,  # Increased epochs
    per_device_train_batch_size=8,  # Increased batch size if GPU memory allows
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,  # Effective batch size = 8 * 2 = 16
    warmup_ratio=0.1,  # 10% of training steps for warmup
    weight_decay=0.01,
    learning_rate=2e-5,  # Slightly lower learning rate
    lr_scheduler_type="cosine",  # Better scheduler
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",
    greater_is_better=True,
    fp16=True,  # Mixed precision training
    save_total_limit=3,
    report_to="none",  # Disable wandb/tensorboard for now
    label_smoothing_factor=0.1,  # Helps with overconfidence
    optim="adamw_torch",  # Better optimizer
    seed=42,
    gradient_checkpointing=True,  # Save memory
    push_to_hub=False
)

# -------------------------------
# 6. Enhanced Metrics
# -------------------------------
def compute_metrics(p):
    logits = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    labels = p.label_ids
    
    # Apply sigmoid to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    
    # Dynamic threshold (can be tuned per class)
    threshold = 0.5
    preds = (probs > threshold).int().numpy()
    
    # Calculate various metrics
    f1_micro = f1_score(y_true=labels, y_pred=preds, average="micro", zero_division=0)
    f1_macro = f1_score(y_true=labels, y_pred=preds, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true=labels, y_pred=preds, average="weighted", zero_division=0)
    
    # Per-class F1 scores
    f1_per_class = f1_score(y_true=labels, y_pred=preds, average=None, zero_division=0)
    
    # Exact match accuracy (all labels must match)
    exact_match = accuracy_score(labels, preds)
    
    # Hamming loss (fraction of wrong labels)
    hamming = np.mean(labels != preds)
    
    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "exact_match": exact_match,
        "hamming_loss": hamming,
    }
    
    # Add per-class F1 scores
    for i, label_name in id2label.items():
        metrics[f"f1_{label_name}"] = f1_per_class[i]
    
    return metrics

# -------------------------------
# 7. Improved Weighted Trainer with Focal Loss option
# -------------------------------
class ImprovedWeightedTrainer(Trainer):
    def __init__(self, *args, pos_weights=None, use_focal_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights
        self.use_focal_loss = use_focal_loss
        
        if self.pos_weights is not None:
            self.pos_weights = self.pos_weights.to(self.args.device)
            
        # Initialize loss function
        if use_focal_loss:
            self.loss_fn = FocalLoss(alpha=self.pos_weights, gamma=2.0)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Move loss function to correct device if needed
        if self.pos_weights is not None and self.pos_weights.device != logits.device:
            self.pos_weights = self.pos_weights.to(logits.device)
            if self.use_focal_loss:
                self.loss_fn = FocalLoss(alpha=self.pos_weights, gamma=2.0).to(logits.device)
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights).to(logits.device)
        
        loss = self.loss_fn(logits, labels.float())
        
        return (loss, outputs) if return_outputs else loss

# -------------------------------
# 8. Focal Loss Implementation (helps with class imbalance)
# -------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

# -------------------------------
# 9. Initialize Trainer with callbacks
# -------------------------------
trainer = ImprovedWeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    pos_weights=pos_weights,
    use_focal_loss=False,  # Set to True to use Focal Loss
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,
            early_stopping_threshold=0.001
        )
    ]
)

# -------------------------------
# 10. Training with evaluation
# -------------------------------
print("🚀 Starting improved model training...")
trainer.train()

# Evaluate on test set
print("\n📊 Evaluating on test set...")
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("\nTest Results:")
for key, value in test_results.items():
    if not key.startswith("eval_"):
        print(f"  {key}: {value:.4f}")

# -------------------------------
# 11. Find optimal thresholds per class
# -------------------------------
print("\n🔍 Finding optimal thresholds per class...")

def find_optimal_thresholds(trainer, dataset, id2label):
    predictions = trainer.predict(dataset)
    logits = predictions.predictions[0] if isinstance(predictions.predictions, tuple) else predictions.predictions
    labels = predictions.label_ids
    
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits)).numpy()
    
    optimal_thresholds = {}
    for i, label_name in id2label.items():
        best_threshold = 0.5
        best_f1 = 0
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            preds = (probs[:, i] > threshold).astype(int)
            f1 = f1_score(labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        optimal_thresholds[label_name] = best_threshold
        print(f"  {label_name}: threshold={best_threshold:.2f}, F1={best_f1:.3f}")
    
    return optimal_thresholds

optimal_thresholds = find_optimal_thresholds(trainer, eval_dataset, id2label)

# -------------------------------
# 12. Save model and configurations
# -------------------------------
model_save_path = "./fine-tuned-analyzer-7labels-improved"
print(f"\n💾 Saving the fine-tuned model to {model_save_path}...")
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Save optimal thresholds
import json
with open(f"{model_save_path}/optimal_thresholds.json", "w") as f:
    json.dump(optimal_thresholds, f, indent=2)

# Save training history
if hasattr(trainer.state, 'log_history'):
    with open(f"{model_save_path}/training_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

print("\n✅ Training complete! Model and configurations saved.")
print(f"📌 Best eval loss: {trainer.state.best_metric:.4f}")
