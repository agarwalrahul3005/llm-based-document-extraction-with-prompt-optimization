import json
import argparse
import dspy

from pathlib import Path

from configs.models import MODELS
from extractors.dspy_extractor import DSPyExtractor


# ---------------------------------
# Configure DSPy
# ---------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model",required=True)
parser.add_argument("--dataset",default="testing")
args = parser.parse_args()

model_name = args.model
model = MODELS[model_name]

lm = dspy.LM(
    f"ollama_chat/{model['ollama_name']}",
    api_base="http://localhost:11434"
)
dspy.configure(lm=lm)

# ---------------------------------
# Paths
# ---------------------------------
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "ocr" / "tesseract"/ args.dataset
OUTPUT_DIR = ROOT / "experiments" / "predictions" / model_name

OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

def main():

    extractor = DSPyExtractor()

    files = sorted(INPUT_DIR.glob("*.json"))

    print(f"\nFound {len(files)} OCR files.\n")

    for index, file in enumerate(files):
        print("\n" + "=" * 100)
        print(f"[{index+1}/{len(files)}] {file.name}")
        print("=" * 100)
        with open(file, encoding="utf-8") as f:
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


if __name__ == "__main__":
    main()