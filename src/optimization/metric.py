import json

from evaluation.matching import (
    flatten_pairs,
    match_pairs,
)


def extraction_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    GEPA optimization metric.

    Parameters
    ----------
    gold : dspy.Example
    pred : dspy.Prediction
    trace : ignored
    pred_name : ignored
    pred_trace : ignored

    Returns
    -------
    float between 0 and 1
    """

    try:
        gt = flatten_pairs(json.loads(gold.response))
        prediction = flatten_pairs(json.loads(pred.response))

    except Exception:
        return 0.0

    matched, _, _ = match_pairs(gt, prediction)

    if len(gt) == 0:
        return 0.0

    return matched / len(gt)