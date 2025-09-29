from text_analyzer import TextAnalyzer

print("--- Initializing Text Analyzer ---")
text_analyzer = TextAnalyzer(device="cpu")

texts = [
    "I am so happy and excited, this is the best day ever!",
    "This is a terrible and frustrating situation, I feel very angry.",
    "I'm not sure how I feel about that.",
    "I feel a bit of sadness and grief after hearing the news.",
    "I should have done better, I always mess things up."
]

print("\n--- Running Predictions ---")
for t in texts:
    result = text_analyzer.predict(t)
    print(f"Input: '{t}'")
    print("All probs:", result["all_probs"])
    print("Final:", result["final"], "\n")

print("--- Test Complete ---")
