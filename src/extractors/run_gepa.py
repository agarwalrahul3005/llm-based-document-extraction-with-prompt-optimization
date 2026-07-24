import argparse
import json

from pathlib import Path

import dspy

from configs.models import MODELS
from dspy_pipeline.module import FormExtractionModule


# --------------------------------------------------------
# Arguments
# --------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",required=True)
args = parser.parse_args()
model_name = args.model
model = MODELS[model_name]


# --------------------------------------------------------
# Configure DSPy
# --------------------------------------------------------
lm = dspy.LM(
    f"ollama_chat/{model['ollama_name']}",
    api_base="http://localhost:11434"
)
dspy.configure(lm=lm)


# --------------------------------------------------------
# Paths
# --------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

OCR_DIR = ROOT / "data" / "ocr" / "tesseract"

PROGRAM_FILE = (
    ROOT
    / "experiments"
    / "gepa"
    / model_name
    / "optimized_program.json"
)

OUTPUT_DIR = (
    ROOT
    / "experiments"
    / "predictions"
    / f"{model_name}_gepa"
)

RAW_DIR = (
    ROOT
    / "experiments"
    / "raw_outputs"
    / f"{model_name}_gepa"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------
# Load optimized program
# --------------------------------------------------------

program = FormExtractionModule()

program.load(PROGRAM_FILE)

print("Loaded optimized program")
print(PROGRAM_FILE)
print()


# --------------------------------------------------------
# Run
# --------------------------------------------------------

files = sorted(OCR_DIR.glob("*.json"))

print(f"Found {len(files)} OCR files\n")


for index, file in enumerate(files):

    print(f"[{index+1}/{len(files)}] {file.name}")

    with open(file, encoding="utf8") as f:
        ocr = json.load(f)

    try:
        prediction = program(document=ocr["words"])
        raw_response = prediction.response

    except Exception as e:
        print(e)
        raw_response = "[]"

    raw_file = RAW_DIR / file.with_suffix(".txt").name

    with open(raw_file, "w", encoding="utf8") as f:
        f.write(raw_response)

    try:
        parsed = json.loads(raw_response)
    except:
        parsed = []

    out_file = OUTPUT_DIR / file.name

    with open(out_file, "w", encoding="utf8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

print()
print("Finished.")