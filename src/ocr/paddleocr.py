from paddleocr import PaddleOCR
import os
import json
from pathlib import Path

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

ROOT = Path(__file__).resolve().parents[2]
input_dir = ROOT / "data" / "training_data" / "images"
output_dir = ROOT / "data" / "ocr_paddleocr"

os.makedirs(output_dir, exist_ok=True)


def run_ocr(image_path):
    result = ocr.ocr(str(image_path))

    words = []

    for line in result[0]:
        bbox = line[0]  # 4 points
        text = line[1][0]

        # Convert bbox to simple format [x1, y1, x2, y2]
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]

        simple_bbox = [
            int(min(x_coords)),
            int(min(y_coords)),
            int(max(x_coords)),
            int(max(y_coords)),
        ]

        words.append({
            "text": text,
            "bbox": simple_bbox
        })

    return {"words": words}


# Run for all images
for file in os.listdir(input_dir):
    print("Input dir:", input_dir)
    print("Files found:", len(list(input_dir.iterdir())))

    if file.endswith(".png") or file.endswith(".jpg"):
        image_path = input_dir / file

        ocr_output = run_ocr(image_path)

        output_file = output_dir / file.replace(".png", ".json")

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(ocr_output, f, indent=2)

print("✅ OCR completed with structured output")