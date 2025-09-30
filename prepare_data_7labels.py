# prepare_data_7labels.py
import pandas as pd
from datasets import load_dataset
import json

# Define the 7 target emotions
TARGET_LABELS = ['joy', 'anger', 'sadness', 'fear', 'love', 'surprise', 'neutral']

def main():
    print("🔹 Loading GoEmotions dataset...")
    dataset = load_dataset("go_emotions", "simplified")

    # Convert to DataFrame
    df = pd.DataFrame(dataset["train"])

    # Initialize binary columns for our 7 target labels
    for label in TARGET_LABELS:
        df[label] = 0.0  # floats to avoid int conversion errors later

    # Get mapping from int IDs → text labels
    id2label = dataset["train"].features["labels"].feature.int2str

    # One-hot encode
    for i, row in df.iterrows():
        for lbl_id in row["labels"]:
            lbl = id2label(lbl_id)
            if lbl in TARGET_LABELS:
                df.at[i, lbl] = 1.0

    # Keep only relevant columns
    df = df[["text"] + TARGET_LABELS]

    # Save dataset in JSONL format
    output_file = "processed_data_7labels.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = {"text": row["text"]}
            record["labels"] = [float(row[lbl]) for lbl in TARGET_LABELS]
            f.write(json.dumps(record) + "\n")

    # Build label maps
    id2label_map = {i: lbl for i, lbl in enumerate(TARGET_LABELS)}
    label2id_map = {lbl: i for i, lbl in id2label_map.items()}

    with open("label_maps_7labels.json", "w") as f:
        json.dump({"id2label": id2label_map, "label2id": label2id_map}, f)

    print(f"✅ Data prepared with {len(TARGET_LABELS)} labels")
    print(f"✅ Saved dataset → {output_file}")
    print("✅ Saved label maps → label_maps_7labels.json")

if __name__ == "__main__":
    main()
