import json
from pathlib import Path
import dspy
from layout.raw_document_builder import RawDocumentBuilder

ROOT = Path(__file__).resolve().parents[2]

OCR_DIR = ROOT / "data" / "ocr" / "easyocr" 
GT_DIR = ROOT / "data" / "ground_truth"


builder = RawDocumentBuilder()

def load_examples(split="train"):
    ocr_dir = OCR_DIR / split
    gt_dir = GT_DIR / split

    examples = []

    files = sorted(gt_dir.glob("*.json"))

    for gt_file in files:

        print(f"Processing {gt_file.name}")

        ocr_file = ocr_dir / gt_file.name

        with open(ocr_file, encoding="utf8") as f:
            ocr = json.load(f)

        with open(gt_file, encoding="utf8") as f:
            gt = json.load(f)

        document = builder.build(ocr["words"])
        example = dspy.Example(
            document=document.to_prompt(),
            response=json.dumps(gt, ensure_ascii=False)
        ).with_inputs("document")

        examples.append(example)

    print(f"\nCreated {len(examples)} DSPy Examples")

    return examples