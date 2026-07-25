import re
from difflib import SequenceMatcher

KEY_MATCH_THRESHOLD = 0.5
VALUE_MATCH_THRESHOLD = 0.7


# ----------------------------------------------------
# NORMALIZATION
# ----------------------------------------------------
def normalize(text):
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ",text)
    return text


# ----------------------------------------------------
# STRING SIMILARITY
# ----------------------------------------------------
def similarity(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


# ----------------------------------------------------
# PAIR MATCH
# ----------------------------------------------------
def is_match(gt, pred):
    return (
        similarity(gt["question"], pred["question"]) >= KEY_MATCH_THRESHOLD
        and
        similarity(gt["answer"], pred["answer"]) >= VALUE_MATCH_THRESHOLD
    )


def flatten_pairs(pairs):
    output = []

    for pair in pairs:
        output.append({
            "question": pair.get("question", ""),
            "answer": pair.get("answer", "")
        })

    return output

# def flatten_pairs(pairs):
#     output = []

#     for pair in pairs:
#         question = pair.get("question", "")
#         answer = pair.get("answer", "")

#         if isinstance(question, dict):
#             question = question.get("text", "")

#         if isinstance(answer, dict):
#             answer = answer.get("text", "")

#         output.append({
#             "question": question,
#             "answer": answer
#         })

#     return output


# ----------------------------------------------------
# MATCH ALL PAIRS
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

    return (matched,missing,incorrect)