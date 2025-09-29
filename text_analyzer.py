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
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze())

        # --- Dynamic thresholding ---
        max_prob = torch.max(probs).item()
        threshold = max(0.5, max_prob * 0.7)  # baseline 0.5, but relative to strongest signal

        results = {}
        for i, prob in enumerate(probs):
            if prob.item() >= threshold:
                label = self.model.config.id2label[i]
                results[label] = round(prob.item(), 2)

        # If no label passes threshold, return the strongest one
        if not results:
            top_idx = torch.argmax(probs).item()
            results[self.model.config.id2label[top_idx]] = round(probs[top_idx].item(), 2)

        return results
