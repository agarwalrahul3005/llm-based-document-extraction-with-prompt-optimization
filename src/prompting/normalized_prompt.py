from prompting.base_prompt import BasePrompt
from layout.normalized_document import NormalizedDocument


class NormalizedPrompt(BasePrompt):

    def build(self,document: NormalizedDocument):
        return f"""
        You are a document information extraction system.
        You receive OCR extracted document lines.
        Each line contains:
            - text
            - normalized bounding box
            - word level coordinates

        Task:     
        1. Extract all key-value pairs.
        2. A key-value pair means:
            - Question: A label describing information.
            - Answer:The value belonging to that label.

        Rules:
        1. Use ONLY text present in the document.
        2. Never create information.
        3. Use layout:
            - values usually appear right of labels
            - values can appear below labels
        4. Return every possible pair once.
        5. Return ONLY JSON.


        Output:

        [
            {{
                "question":
                    {{
                        "text":"",
                        "bbox":[]
                    }},

                "answer":
                    {{
                        "text":"",
                        "bbox":[]
                    }}
            }}
        ]

        DOCUMENT
        {document.to_prompt()}

        """
