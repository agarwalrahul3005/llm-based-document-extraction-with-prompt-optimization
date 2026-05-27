import json
from pathlib import Path

gt_dir = Path("data/ground_truth")
pred_dir = Path("data/predictions")

for file in gt_dir.glob("*.json"):
    gt = json.load(open(file))
    pred = json.load(open(pred_dir / file.name))

    print(file.name, len(gt), len(pred))