import pandas as pd
from datasets import load_dataset
import json
from collections import Counter

TARGET_LABELS = ['joy', 'anger', 'sadness', 'fear', 'love', 'surprise', 'neutral']

def main():
    print("🔹 Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", "simplified")

    df = pd.DataFrame(dataset["train"])

    # Initialize binary columns
    for label in TARGET_LABELS:
        df[label] = 0.0

    id2label = dataset["train"].features["labels"].feature.int2str

    # Fill one-hot labels
    for i, row in df.iterrows():
        for lbl_id in row["labels"]:
            lbl = id2label(lbl_id)
            if lbl in TARGET_LABELS:
                df.at[i, lbl] = 1.0

    # Keep only text + target labels
    df = df[["text"] + TARGET_LABELS]

    # Drop rows with all zeros
    df = df[df[TARGET_LABELS].sum(axis=1) > 0]

    print(f"🔹 After filtering: {len(df)} rows remain.")

    # Check distribution
    label_counts = df[TARGET_LABELS].sum().to_dict()
    print("🔹 Label distribution:", label_counts)

    # Save JSONL
    output_file = "processed_data_7labels.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {"text": row["text"]}
            record["labels"] = [row[lbl] for lbl in TARGET_LABELS]
            f.write(json.dumps(record) + "\n")

    print(f"✅ Data prepared! Saved to {output_file}")

if __name__ == "__main__":
    main()
