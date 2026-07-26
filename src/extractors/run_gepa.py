import argparse
import json
from pathlib import Path

import dspy

from configs.models import MODELS
from extractors.dspy_extractor import DSPyExtractor


# ----------------------------------------------------
# Arguments
# ----------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument("--model",required=True,)
args = parser.parse_args()
model_name = args.model
model = MODELS[model_name]


# ----------------------------------------------------
# Configure LM
# ----------------------------------------------------
lm = dspy.LM(
    f"ollama_chat/{model['ollama_name']}",
    api_base="http://localhost:11434",
)
dspy.configure(lm=lm)


# ----------------------------------------------------
# Paths
# ----------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]

OCR_DIR = ROOT / "data"/ "ocr" / "easyocr"/ "testing"
OUTPUT_DIR = ROOT / "experiments" / "predictions" / f"{model_name}_gepa"
PROGRAM_FILE = ROOT / "experiments" / "gepa"  / model_name / f"{model_name}_optimized_program.json"


OUTPUT_DIR.mkdir( parents=True,exist_ok=True)


# ----------------------------------------------------
# Load optimized extractor
# ----------------------------------------------------
extractor = DSPyExtractor(optimized_program=PROGRAM_FILE)


# ----------------------------------------------------
# Process files
# ----------------------------------------------------
files = sorted(OCR_DIR.glob("*.json"))

print(f"Processing {len(files)} OCR files\n")

for index, file in enumerate(files):
    print("\n" + "=" * 100)
    print(f"[{index+1}/{len(files)}] {file.name}")

    with open(file, encoding="utf8") as f:
        ocr = json.load(f)

        try:
            result = extractor.extract(ocr["words"],file.name)
            prediction = result["parsed"]
            print(f"Predicted {len(prediction)} pairs using {model_name}")
            print("=" * 100 + "\n")
        except Exception as e:
            print(file.name)
            print(e)
            prediction = []    

    output_file = OUTPUT_DIR / file.name
    with open( output_file, "w", encoding="utf-8") as f:
        json.dump(prediction, f, indent=2, ensure_ascii=False)    

print("\nExtraction Finished.")