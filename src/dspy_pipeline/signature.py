import dspy


class ExtractForm(dspy.Signature):
    """
        You are an expert in extracting structured information from scanned forms.
        The input consists of OCR text lines.

        Identify all KEY-VALUE pairs: 
        - A short label (question) 
        - Followed by its corresponding value (answer) 

        Rules
        • Use only OCR text.
        • Never invent fields.
        • Preserve OCR spelling.
        • Questions usually appear to the left or above answers.
        • Answers may appear mostly right or below of labels.
        • Return only valid JSON.
    """

    document = dspy.InputField(desc="OCR document with layout information.")

    # response = dspy.OutputField(desc="JSON array containing question-answer pairs.")
    response = dspy.OutputField(
        desc="""
            JSON array
            [
                {
                    "question":"...",
                    "answer":"..."
                }
            ]
            """
    )