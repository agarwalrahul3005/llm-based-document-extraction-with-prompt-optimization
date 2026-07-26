import argparse
from pathlib import Path

from evaluation.evaluate import evaluate



parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--dataset", default="testing")
args = parser.parse_args()

ROOT = Path(__file__).resolve().parents[2]
groundtruth_dir = ROOT / "data" / "ground_truth" / args.dataset
prediction_dir = ROOT / "experiments" / "predictions" / f"{args.model}_gepa"
output_file = ROOT / "experiments" / "results" / f"{args.model}_gepa.json"


evaluate(gt_dir=groundtruth_dir, prediction_dir=prediction_dir,output_file=output_file)