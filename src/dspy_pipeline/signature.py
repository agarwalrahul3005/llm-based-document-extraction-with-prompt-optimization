import dspy


class ExtractForm(dspy.Signature):
    """
    Extract all question-answer pairs from a document.

    OCR may contain spelling mistakes.

    Use layout information to determine relationships.

    Return ONLY valid JSON.

    Do not hallucinate.
    """

    document = dspy.InputField(
        desc="Document representation with text lines and bounding boxes."
    )

    response = dspy.OutputField(
        desc="""
        Return ONLY a JSON array.

        Example:

        [
        {
            "question": {
            "text": "COMPOUND",
            "bbox": [84,109,136,119]
            },
            "answer": {
            "text": "3-Hydroxy-3-methylbutanoic acid",
            "bbox": [145,98,507,116]
            }
        }
        ]

        Do not return explanations.

        Do not wrap in markdown.

        Do not return any other field.

        Return ONLY JSON.
        """
    )