import dspy


class ExtractForm(dspy.Signature):
    """
        You are an expert in document understanding.
        The input is an OCR document represented as text lines.
        Every line contains:line text, bounding box, words

        Your task is to extract every question-answer pair.

        Rules
        1. Use only text present in the OCR.
        2. Never invent text.
        3. Preserve OCR spelling.
        4. Extract every pair.
        5. Labels usually appear left of values.
        6. Values may appear below labels.
        7. Return ONLY valid JSON.

        Output format
        [
            {
                "question":"...",
                "answer":"..."
            }
        ] 
    """

    document = dspy.InputField(desc="OCR document with layout information.")

    response = dspy.OutputField(desc="JSON array containing question-answer pairs.")