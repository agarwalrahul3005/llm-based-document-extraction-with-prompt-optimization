import json
from pathlib import Path
import dspy
from layout.raw_document_builder import RawDocumentBuilder

ROOT = Path(__file__).resolve().parents[2]

OCR_DIR = ROOT / "data" / "ocr" / "tesseract" 
GT_DIR = ROOT / "data" / "ground_truth" / "training"


builder = RawDocumentBuilder()

def load_examples():
    examples = []

    files = sorted(GT_DIR.glob("*.json"))

    print(f"\nLoading {len(files)} training examples\n")

    for gt_file in files:

        print(f"\nProcessing {gt_file.name}")

        ocr_file = OCR_DIR / gt_file.name

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