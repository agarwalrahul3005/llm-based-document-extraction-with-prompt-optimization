import pytesseract
import cv2
import os
import json
import argparse
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--dataset",default="train")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]

input_dir = ROOT / "data" / "funsd" / args.dataset / "images"
output_dir = ROOT / "data" / "ocr" / "tesseract" / args.dataset

os.makedirs(output_dir, exist_ok=True)


def run_ocr(image_path):
    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError("Unable to read image")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0,  255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    custom_config = r'--oem 3 --psm 6'
    data = pytesseract.image_to_data(gray, config=custom_config, output_type=pytesseract.Output.DICT)

    words = []

    n = len(data['text'])
    for i in range(n):
        text = data['text'][i].strip()
        if not text:
            continue

        conf = float(data["conf"][i])
        if conf < 30:
            continue

         
        x, y, w, h = (
            data['left'][i],
            data['top'][i],
            data['width'][i],
            data['height'][i]
        )

        words.append({
            "text": text,
            "bbox": [x, y, x + w, y + h],
            "confidence": conf
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

print("✅ OCR completed (Tesseract)") 