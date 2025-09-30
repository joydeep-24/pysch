# test_analyzer.py
from text_analyzer import TextAnalyzer

def main():
    model = TextAnalyzer(model_path="./fine-tuned-analyzer-7labels")

    test_sentences = [
        "I am very happy today!",
        "I feel so sad and hopeless.",
        "I am terrified of tomorrow.",
        "I really love spending time with my family.",
        "That surprise party was amazing!",
        "I am angry about what happened.",
        "I'm just feeling okay, nothing special."
    ]

    for sent in test_sentences:
        preds = model.predict(sent)
        print(f"\nInput: {sent}")
        print("Predictions:", preds)

if __name__ == "__main__":
    main()
