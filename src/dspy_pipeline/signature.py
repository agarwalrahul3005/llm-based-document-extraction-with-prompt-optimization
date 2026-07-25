import dspy


class ExtractForm(dspy.Signature):
    """
        You are an expert in document understanding.
        The input is an OCR document represented as text lines.
        Every line contains:line text, bounding box, words
        Your task is to extract every question-answer pair.

        Rules
        1. use only text present in the OCR., Label(question) and Value(answer) must be text present in OCR.
        2. Preserve OCR spelling.
        3. Extract every pair.
        4. Labels usually appear left of values.
        5. Values may appear below labels or right of labels.
        6. Return ONLY valid JSON.

        Output format to follow:
        [
            {
                "question":"...",
                "answer":"..."
            }
        ] 
    """

    document = dspy.InputField(desc="OCR document with layout information.")

    response = dspy.OutputField(desc="JSON array containing question-answer pairs.")