import os
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
GT_DIR = ROOT / "data" / "ground_truth"


def is_valid_bbox(bbox):
    return (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(isinstance(x, (int, float)) for x in bbox)
    )


def validate_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        return False, "Not a list"

    for i, pair in enumerate(data):
        if "question" not in pair or "answer" not in pair:
            return False, f"Missing keys at index {i}"

        q = pair["question"]
        a = pair["answer"]

        if not q.get("text") or not a.get("text"):
            return False, f"Empty text at index {i}"

        if not is_valid_bbox(q.get("bbox")):
            return False, f"Invalid question bbox at index {i}"

        if not is_valid_bbox(a.get("bbox")):
            return False, f"Invalid answer bbox at index {i}"

    return True, "OK"


def main():
    files = list(GT_DIR.glob("*.json"))

    if not files:
        print("❌ No ground truth files found")
        return

    valid_count = 0

    for file in files:
        is_valid, msg = validate_file(file)

        if not is_valid:
            print(f"❌ {file.name}: {msg}")
        else:
            valid_count += 1

    print(f"\n✅ {valid_count}/{len(files)} files valid")


if __name__ == "__main__":
    main()