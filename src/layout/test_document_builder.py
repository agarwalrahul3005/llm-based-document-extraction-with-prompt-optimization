import json
from pathlib import Path

from src.layout.raw_document_builder import RawDocumentBuilder


ROOT = Path(__file__).resolve().parents[2]

ocr_file = ROOT / "data" / "ocr" / "tesseract" / "00040534.json"

with open(ocr_file) as f:
    ocr = json.load(f)

builder = RawDocumentBuilder()

document = builder.build(
    ocr["words"]
)

print("=" * 60)

print(document.to_prompt())

print("=" * 60)