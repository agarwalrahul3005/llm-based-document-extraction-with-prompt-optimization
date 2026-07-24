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
    "ollama_chat/qwen2.5:7b",
    api_base="http://localhost:11434",
    temperature=1.0
)

dspy.configure(lm=lm)

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

print("=" * 80)
print("Loading training dataset...")
print("=" * 80)

trainset = load_examples()

print(f"Loaded {len(trainset)} training examples\n")


# ----------------------------------------------------
# Baseline Program
# ----------------------------------------------------
print("=" * 80)
print("Creating baseline DSPy program...")
print("=" * 80)

program = FormExtractionModule()


# ----------------------------------------------------
# GEPA Optimizer
# ----------------------------------------------------
print("=" * 80)
print("Initializing GEPA...")
print("=" * 80)

optimizer = dspy.GEPA(metric=extraction_metric, reflection_lm=reflection_lm, max_full_evals=2)

# ----------------------------------------------------
# Optimize
# ----------------------------------------------------
print("=" * 80)
print("Running optimization...")
print("=" * 80)

optimized_program = optimizer.compile(student=program, trainset=trainset)

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

optimized_program.save(PROGRAM_FILE)

print()
print("=" * 80)
print("Optimization Complete")
print("=" * 80)
print(f"Saved to:\n{PROGRAM_FILE}")