# test_analyzer.py
from text_analyzer import TextAnalyzer

# --- A list of different sentences to test the model ---
test_sentences = [
    "I am so happy and excited, this is the best day ever!",
    "This is a terrible and frustrating situation, I feel very angry.",
    "I'm not sure how I feel about that.",
    "I feel a bit of sadness and grief after hearing the news.",
    "I should have done better, I always mess things up."
]

print("--- Initializing Text Analyzer ---")
# This will load your fine-tuned model from Google Drive
# Make sure your Drive is mounted
text_analyzer = TextAnalyzer()
print("--------------------------------\n")


print("--- Running Predictions ---")
# Loop through each sentence and print the model's prediction
for sentence in test_sentences:
    prediction = text_analyzer.predict(sentence)
    print(f"Input: '{sentence}'")
    print(f"Output: {prediction}\n")

print("--- Test Complete ---")