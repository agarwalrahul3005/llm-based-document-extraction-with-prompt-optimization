import json
from evaluation.matching import match_pairs

# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------
def evaluate(gt_dir, prediction_dir, output_file):
    overall_gt = 0
    overall_pred = 0
    overall_match = 0
    report = []

    files = sorted(gt_dir.glob("*.json"))

    print(f"\nEvaluating {len(files)} files\n")

    for gt_file in files:

        pred_file = prediction_dir / gt_file.name

        if not pred_file.exists():
            print(f"Missing prediction : {gt_file.name}")
            continue

        # gt = flatten_pairs(json.load(open(gt_file)))
        # pred = flatten_pairs(json.load(open(pred_file)))

        print("-------------------------------------------------------------------------------")
        print(f"Evaluating {gt_file.name}")
        with open(gt_file, encoding="utf-8") as f:
            gt = json.load(f)

        with open(pred_file, encoding="utf-8") as f:
            pred = json.load(f)

        matched, missing, incorrect = match_pairs(gt, pred)

        precision = matched / len(pred) if pred else 0
        recall = matched / len(gt) if gt else 0
        f1 = ((2 * precision * recall) /(precision + recall)) if precision + recall else 0
        
        overall_gt += len(gt)
        overall_pred += len(pred)
        overall_match += matched

        report.append({
            "file": gt_file.name,
            "ground_truth": len(gt),
            "predictions": len(pred),
            "matched": matched,
            "precision": round(precision,3),
            "recall": round(recall,3),
            "f1": round(f1,3)
        })

        print(
            f"GT={len(gt)} "
            f"PRED={len(pred)} "
            f"MATCH={matched}"
        )

        print(
            f"Precision={precision:.2f}"
            f" Recall={recall:.2f}"
            f" F1={f1:.2f}"
        )

    overall_precision = overall_match / overall_pred if overall_pred else 0
    overall_recall =  overall_match / overall_gt if overall_gt else 0
    overall_f1 = (
        2 * overall_precision * overall_recall /

        (overall_precision + overall_recall)

        if overall_precision + overall_recall

        else 0
    )

    summary = {
        "overall": {
            "ground_truth": overall_gt,
            "predictions": overall_pred,
            "matched": overall_match,
            "precision": round(overall_precision,3),
            "recall": round(overall_recall,3),
            "f1": round(overall_f1,3)
        },
        "files": report
    }

    with open( output_file, "w",encoding="utf-8") as f: json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n===========================================================")
    print("OVERALL")
    print(json.dumps(summary["overall"], indent=4))
    print()
    print(f"Saved to {output_file}")
    print("==============================================================")