import json
from pathlib import Path
from difflib import SequenceMatcher


ROOT = Path(__file__).resolve().parents[2]

GT_PATH = ROOT / "data" / "ground_truth" / "training"
PRED_PATH = ROOT / "data" / "predictions" / "prompt_v3_llama3"

OUTPUT_DIR = ROOT / "experiments" / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "prompt_v3_llama3.json"

MATCH_THRESHOLD = 0.55


# ----------------------------------------------------
# Helpers
# ----------------------------------------------------

def normalize(text):
    if text is None:
        return ""
    return str(text).strip().lower()


def similarity(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def flatten_pairs(pairs):
    output = []

    for pair in pairs:
        question = pair.get("question", "")
        answer = pair.get("answer", "")

        if isinstance(question, dict):
            question = question.get("text", "")

        if isinstance(answer, dict):
            answer = answer.get("text", "")

        output.append({
            "question": question,
            "answer": answer
        })

    return output


def is_match(gt, pred):
    return similarity( gt["question"], pred["question"]) >= MATCH_THRESHOLD and similarity( gt["answer"],pred["answer"]) >= MATCH_THRESHOLD
 

# ----------------------------------------------------
# Matching
# ----------------------------------------------------

def match_pairs(gt_pairs, pred_pairs):
    matched = 0
    used_predictions = set()
    missing = []
    incorrect = []

    for gt in gt_pairs:
        found = False

        for index, pred in enumerate(pred_pairs):
            if index in used_predictions:
                continue

            if is_match(gt, pred):
                matched += 1
                used_predictions.add(index)
                found = True
                break

        if not found:
            missing.append(gt)

    for index, pred in enumerate(pred_pairs):
        if index not in used_predictions:
            incorrect.append(pred)

    return matched, missing, incorrect


# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------

def evaluate():

    overall_gt = 0
    overall_pred = 0
    overall_match = 0
    report = []

    files = sorted(GT_PATH.glob("*.json"))

    print(f"\nEvaluating {len(files)} files\n")

    for gt_file in files:

        pred_file = PRED_PATH / gt_file.name

        if not pred_file.exists():
            print(f"Missing prediction : {gt_file.name}")
            continue

        gt = flatten_pairs(json.load(open(gt_file)))

        pred = flatten_pairs(
            json.load(open(pred_file))
        )

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

        print("--------------------------------")
        print(gt_file.name)
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

    with open( OUTPUT_FILE, "w",encoding="utf-8") as f: json.dump(summary, f, indent=4, ensure_ascii=False)

    print("\n==============================")
    print("OVERALL")
    print("==============================")
    print(json.dumps(summary["overall"], indent=4))
    print()
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate()