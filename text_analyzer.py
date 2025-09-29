# text_analyzer.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextAnalyzer:
    def __init__(self, model_path="./fine-tuned-analyzer"):
        """
        Loads the fine-tuned Longformer model and tokenizer.
        """
        print(f"🔹 Loading Text Analyzer model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print("✅ Text Analyzer loaded.")

    def predict(self, text):
        """
        Analyzes a single piece of text.
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze())
        predictions = (probs > 0.5).int()

        results = {}
        for i, is_present in enumerate(predictions):
            if is_present:
                label = self.model.config.id2label[i]
                results[label] = round(probs[i].item(), 2)
        
        return results if results else {"main_finding": "Neutral"}