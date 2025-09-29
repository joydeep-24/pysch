# text_analyzer.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class TextAnalyzer:
    def __init__(self, model_path="/content/drive/MyDrive/fine-tuned-analyzer"):
        print(f"🔹 Loading Text Analyzer model from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model directory not found at {model_path}."
            )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("✅ Text Analyzer loaded.")
    def predict(self, text):
    # Tokenize input
    inputs = self.tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )

    # Forward pass (no gradients)
    with torch.no_grad():
        logits = self.model(**inputs).logits

    # Use softmax for single-label classification
    probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()

    # Pick the top class
    top_idx = torch.argmax(probs).item()
    label = self.model.config.id2label[top_idx]
    confidence = round(probs[top_idx].item(), 2)

    return {label: confidence}
