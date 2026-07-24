import json
import argparse
import dspy

from pathlib import Path

from configs.models import MODELS
from dspy_pipeline.extractor import DSPyExtractor


# ---------------------------------
# Configure DSPy
# ---------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--model",required=True)

# parser.add_argument("--dataset",default="training")

args = parser.parse_args()
model_name = args.model
model = MODELS[model_name]

lm = dspy.LM(
    f"ollama_chat/{model['ollama_name']}",
    api_base="http://localhost:11434"
)

dspy.configure(lm=lm)

metadata = {
    "experiment": model_name,
    "ollama_model": model["ollama_name"],
    "framework": "DSPy",
    "ocr": "Tesseract",
    "representation": "Line + Bounding Box",
    "prompt": "BaselinePrompt"
}

# ---------------------------------
# Paths
# ---------------------------------

ROOT = Path(__file__).resolve().parents[2]

# INPUT_DIR = ROOT/ "data" / "ocr" / "tesseract"/ args.dataset
INPUT_DIR = ROOT / "data" / "ocr" / "tesseract"
OUTPUT_DIR = ROOT / "experiments" / "predictions" / model_name
RAW_OUTPUT_DIR = ROOT / "experiments" / "raw_outputs" / model_name
REGISTRY = ROOT / "experiments" / "experiment_registry.json"



OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
RAW_OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

registry = {}

if REGISTRY.exists():
    with open(REGISTRY) as f:
        registry = json.load(f)

registry[model_name] = metadata

with open(REGISTRY, "w") as f:
    json.dump(registry, f, indent=2)



# ---------------------------------
# Main
# ---------------------------------

def main():

    extractor = DSPyExtractor(prompt_name="baseline")

    files = sorted(
        INPUT_DIR.glob("*.json")
    )

    print(f"\nFound {len(files)} OCR files.\n")

    for index, file in enumerate(files):
        print(f"[{index+1}/{len(files)}] {file.name}")

        with open(file, encoding="utf-8") as f:
            ocr = json.load(f)

        try:
            result = extractor.extract(ocr["words"],file.name)
            prompt = result["prompt"]
            raw_output = result["raw"]
            prediction = result["parsed"]
            print(f"Predicted {len(prediction)} pairs using {model_name}")
            print("=" * 100 + "\n")
        except Exception as e:
            print(file.name)
            print(e)
            prediction = []
            prompt = ""
            raw_output = str(e)

        output_file = OUTPUT_DIR / file.name
        raw_file = RAW_OUTPUT_DIR / file.with_suffix(".txt").name

        with open( output_file, "w", encoding="utf-8") as f:
            json.dump(prediction, f, indent=2, ensure_ascii=False)

        with open(raw_file, "w", encoding="utf-8") as f:
            f.write("=" * 80)
            f.write("\nPROMPT\n")
            f.write("=" * 80)
            f.write("\n\n")

            f.write(prompt)

            f.write("\n\n")

            f.write("=" * 80)
            f.write("\nRESPONSE\n")
            f.write("=" * 80)
            f.write("\n\n")

            f.write(raw_output)

    print("\nExtraction Finished.")


if __name__ == "__main__":

    main()