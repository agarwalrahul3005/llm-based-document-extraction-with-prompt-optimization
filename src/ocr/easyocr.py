import easyocr
import cv2
import os
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="train")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]

input_dir = ROOT / "data" / "funsd" / args.dataset / "images"
output_dir = ROOT / "data" / "ocr" / "easyocr" / args.dataset

os.makedirs(output_dir, exist_ok=True)

# Initialize EasyOCR once
reader = easyocr.Reader(['en'],gpu=False)


def run_ocr(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Unable to read image: {image_path}")

    results = reader.readtext(img)

    words = []

    for bbox, text, confidence in results:

        if not text.strip():
            continue

        if confidence < 0.30:
            continue

        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]

        x1 = int(min(xs))
        y1 = int(min(ys))
        x2 = int(max(xs))
        y2 = int(max(ys))

        words.append({
            "text": text.strip(),
            "bbox": [x1, y1, x2, y2],
            "confidence": confidence
        })

    return {"words": words}


for file in os.listdir(input_dir):

    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = input_dir / file

        try:
            ocr_output = run_ocr(image_path)
            output_file = output_dir / (Path(file).stem + ".json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(ocr_output, f, indent=2)
        except Exception as e:
            print(f"Error on {file}: {e}")
            continue

print("✅ OCR completed (EasyOCR)")