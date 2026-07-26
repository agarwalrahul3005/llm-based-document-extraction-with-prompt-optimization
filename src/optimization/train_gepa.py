import argparse
from pathlib import Path

import dspy

from configs.models import MODELS
from dspy_pipeline.module import FormExtractionModule
from optimization.build_dataset import load_examples
from optimization.metric import extraction_metric


# ----------------------------------------------------
# Arguments
# ----------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Model name from configs.models")
args = parser.parse_args()
model_name = args.model
if model_name not in MODELS:
    raise ValueError(f"Unknown model: {model_name}")
model = MODELS[model_name]

# ----------------------------------------------------
# Configure DSPy
# ----------------------------------------------------
lm = dspy.LM(
    f"ollama_chat/{model['ollama_name']}",
    api_base="http://localhost:11434"
)

reflection_lm = dspy.LM(
    "ollama_chat/llama3",
    api_base="http://localhost:11434",
    temperature=0.5
)

dspy.configure(lm=lm)

# ----------------------------------------------------
# Load Datasets
# ----------------------------------------------------
print("=" * 50)
print("Loading training dataset...")
trainset = load_examples("train")
print(f"Loaded {len(trainset)} training examples\n")
print()
print("Loading validation dataset...")
valset = load_examples("validation")
print(f"Loaded {len(valset)} validation examples\n")
print("=" * 50)

# ----------------------------------------------------
# Baseline Program
# ----------------------------------------------------
program = FormExtractionModule()


# ----------------------------------------------------
# GEPA Optimizer Configuration
# ----------------------------------------------------
print("Initializing GEPA...\n")
optimizer = dspy.GEPA(
    metric=extraction_metric,
    reflection_lm=reflection_lm,
    max_full_evals=4,
    num_threads=1
)

# ----------------------------------------------------
# Optimize
# ----------------------------------------------------
print("Running optimization...\n")
optimized_program = optimizer.compile(student=program, trainset=trainset, valset=valset)

# ----------------------------------------------------
# Save
# ----------------------------------------------------
OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "experiments"
    / "gepa"
    / model_name
)

OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

PROGRAM_FILE = OUTPUT_DIR / f"{model_name}_optimized_program.json"

optimized_program.save(str(PROGRAM_FILE))

print("Optimization Complete\n")
print(f"Saved to:\n{PROGRAM_FILE}")
print("=" * 50)