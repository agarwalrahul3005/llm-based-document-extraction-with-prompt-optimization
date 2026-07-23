from layout.document import Document
from prompting.base_prompt import BasePrompt


class PromptV2(BasePrompt):

    def build(self, document: Document):

        prompt = f"""
        You are an expert in document understanding.

        The input is an OCR document that has already been grouped into text lines.

        Each line contains:
            - Line text
            - Line bounding box
            - Individual OCR words
            - Word bounding boxes

        Your task is to extract ALL key-value pairs.

        Definition:
        A key-value pair consists of:
            - Question (Label)
            - Answer (Value)

        Guidelines:
        1. Use only text present in the document.
        2. Do not invent keys or values.
        3. Never invent text.
        4. Extract every key-value pair
        5. Labels often appear on the left.
        6. Values usually appear to the right or directly below.
        7. OCR may contain spelling mistakes.
        8. Return ONLY valid JSON.

        Output Format:
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
        
        OCR Document Representation
        --------------------------------
        {document.to_prompt()}
        """

        return prompt