# text_analyzer.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

class TextAnalyzer:
    def __init__(self, model_path="./fine-tuned-analyzer-7labels", threshold=0.4):
        """
        Loads the fine-tuned Longformer model for 7-label classification.
        """
        print(f"🔹 Loading Text Analyzer from {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Mount Google Drive if needed.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # inference mode
        self.threshold = threshold
        print("✅ Text Analyzer loaded.")

    def predict(self, text: str):
        """
        Predicts emotions for a given text using dynamic thresholding.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        results = {}

        for i, p in enumerate(probs):
            label = self.model.config.id2label[i]
            if p >= self.threshold:
                results[label] = round(float(p), 3)

        return results if results else {"main_finding": "Neutral"}
