import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "training_data" / "annotations"
OUTPUT_DIR = ROOT / "data" / "ground_truth" / "training"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    form = data.get("form", [])
    id_map = {item["id"]: item for item in form}

    pairs = []

    for item in form:
        # Only process questions
        if item.get("label") != "question":
            continue

        q_id = item["id"]
        q_text = item.get("text", "").strip()
        q_box = item.get("box", [])

        if not q_text or not q_box:
            continue

        links = item.get("linking", [])

        for link in links:
            if len(link) != 2:
                continue

            id1, id2 = link

            # Ensure correct direction
            if id1 == q_id:
                a_id = id2
            elif id2 == q_id:
                a_id = id1
            else:
                continue

            answer_item = id_map.get(a_id)

            if not answer_item:
                continue

            if answer_item.get("label") != "answer":
                continue

            a_text = answer_item.get("text", "").strip()
            a_box = answer_item.get("box", [])

            if not a_text or not a_box:
                continue

            pair = {
                "question": {
                    "text": q_text,
                    "bbox": q_box
                },
                "answer": {
                    "text": a_text,
                    "bbox": a_box
                }
            }

            pairs.append(pair)

    return pairs


def main():
    files = list(INPUT_DIR.glob("*.json"))
    count = 0

    for file in files:
        pairs = convert_file(file)

        if not pairs:
            continue

        output_file = OUTPUT_DIR / file.name

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(pairs, f, indent=2)

        count += 1

    print(f"✅ Converted {count} files with valid KV pairs")


if __name__ == "__main__":
    main()