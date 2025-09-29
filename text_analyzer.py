import torch
from torch.nn.functional import sigmoid
from transformers import LongformerTokenizer, LongformerForSequenceClassification

class TextAnalyzer:
    def __init__(self, model_name="allenai/longformer-base-4096", device="cpu"):
        self.device = device
        print(f"✅ Loading model: {model_name} on {device}...")

        # Load tokenizer & model
        self.tokenizer = LongformerTokenizer.from_pretrained(model_name)
        self.model = LongformerForSequenceClassification.from_pretrained(model_name).to(device)

        # Define your labels (adjust if you trained with different ones)
        self.labels = ["Positive", "Negative"]

    def predict(self, text: str):
        # Encode
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        ).to(self.device)

        # Forward pass
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Sigmoid for multilabel
        probs = sigmoid(logits).cpu().numpy()[0]

        # Map labels → probabilities
        predictions = {label: float(prob) for label, prob in zip(self.labels, probs)}

        # Final: choose top label
        top_idx = probs.argmax()
        final_label = self.labels[top_idx]

        return {"all_probs": predictions, "final": {final_label: float(probs[top_idx])}}
