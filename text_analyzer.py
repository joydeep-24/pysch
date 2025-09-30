# text_analyzer.py
import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextAnalyzer:
    def __init__(self, model_path="/content/drive/MyDrive/fine-tuned-analyzer-7labels"):
        """
        Loads the fine-tuned Longformer model, tokenizer, and per-class thresholds.
        """
        print(f"🔹 Loading Text Analyzer from {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

        # Load thresholds if available, else use default 0.5
        thresholds_path = os.path.join(model_path, "optimal_thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                self.thresholds = json.load(f)
            print(f"✅ Loaded per-class thresholds: {self.thresholds}")
        else:
            self.thresholds = {label: 0.5 for label in self.model.config.id2label.values()}
            print("⚠️ No thresholds found, using default 0.5 for all labels.")

        print("✅ Text Analyzer loaded.")

    def predict(self, text):
        """
        Analyze text and return labels with probabilities.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze()).cpu().numpy()

        results = {}
        for i, prob in enumerate(probs):
            label = self.model.config.id2label[i]
            threshold = self.thresholds.get(label, 0.5)
            if prob >= threshold:
                results[label] = round(float(prob), 3)

        return results if results else {"main_finding": "Neutral"}
