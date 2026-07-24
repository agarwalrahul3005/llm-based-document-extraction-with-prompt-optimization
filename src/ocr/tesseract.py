import pytesseract
import cv2
import os
import json
from pathlib import Path

print(pytesseract.get_tesseract_version())

# IMPORTANT: Set this if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ROOT = Path(__file__).resolve().parents[2]
input_dir = ROOT / "data" / "training_data" / "images"
output_dir = ROOT / "data" / "ocr" / "tesseract" / "train"

os.makedirs(output_dir, exist_ok=True)


def run_ocr(image_path):
    img = cv2.imread(str(image_path))

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    words = []

    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()

        if text:  # ignore empty
            x, y, w, h = (
                data['left'][i],
                data['top'][i],
                data['width'][i],
                data['height'][i]
            )

            words.append({
                "text": text,
                "bbox": [x, y, x + w, y + h]
            })

    return {"words": words}


for file in os.listdir(input_dir):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        image_path = input_dir / file

        try:
            ocr_output = run_ocr(image_path)

            output_file = output_dir / file.replace(".png", ".json")

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(ocr_output, f, indent=2)

        except Exception as e:
            print(f"Error on {file}: {e}")
            continue

print("✅ OCR completed (Tesseract)") 