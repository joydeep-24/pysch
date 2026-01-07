# text_analyzer.py
import torch
import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Fixed emotion order expected by fusion model
EMOTION_ORDER = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

class TextAnalyzer:
    def __init__(self, model_path="/content/drive/MyDrive/fine-tuned-analyzer-7labels"):
        """
        Loads the fine-tuned transformer-based emotion classifier.
        """
        print(f"🔹 Loading Text Analyzer from {model_path}...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

        # Load thresholds if available (used only for human-readable output)
        thresholds_path = os.path.join(model_path, "optimal_thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, "r") as f:
                self.thresholds = json.load(f)
            print(f"✅ Loaded per-class thresholds.")
        else:
            self.thresholds = {label: 0.5 for label in self.model.config.id2label.values()}
            print("⚠️ No thresholds found, using default 0.5.")

        print("✅ Text Analyzer loaded.")

    # ---------------------------------------------------------------------
    # ORIGINAL METHOD (UNCHANGED – FOR DEMO / READABLE OUTPUT)
    # ---------------------------------------------------------------------
    def predict(self, text):
        """
        Returns human-readable emotion labels with probabilities
        (thresholded).
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze()

        probs = torch.sigmoid(logits).cpu().numpy()

        results = {}
        for i, prob in enumerate(probs):
            label = self.model.config.id2label[i]
            threshold = self.thresholds.get(label, 0.5)
            if prob >= threshold:
                results[label] = round(float(prob), 3)

        return results if results else {"main_finding": "Neutral"}

    # ---------------------------------------------------------------------
    # NEW METHOD (FOR FUSION MODEL INPUT)
    # ---------------------------------------------------------------------
    def predict_vector(self, text):
        """
        Returns a fixed-size 7D emotion probability vector
        in the order expected by the fusion model.

        Output shape: (7,)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits.squeeze()

        probs = torch.sigmoid(logits).cpu().numpy()

        emotion_vector = np.zeros(len(EMOTION_ORDER), dtype=np.float32)

        for i, label in self.model.config.id2label.items():
            if label in EMOTION_ORDER:
                idx = EMOTION_ORDER.index(label)
                emotion_vector[idx] = probs[i]

        return emotion_vector
