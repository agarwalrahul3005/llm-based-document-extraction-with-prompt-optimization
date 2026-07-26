import json
import re
import dspy
from pathlib import Path

from layout.raw_document_builder import RawDocumentBuilder
from dspy_pipeline.module import FormExtractionModule


class DSPyExtractor:

    def __init__(self, optimized_program=None):
        self.builder = RawDocumentBuilder()
        self.module = FormExtractionModule()
        if optimized_program:
            optimized_program = Path(optimized_program)
            print(optimized_program)
            self.module.load(str(optimized_program))

    def parse_json(self, text):
        # -------------------------------------------------
        # Case 1: Response is already valid JSON
        # -------------------------------------------------
        try:
            return json.loads(text)
        except Exception:
            pass

        # -------------------------------------------------
        # Case 2: Extract JSON array from response
        # -------------------------------------------------
        array_match = re.search(r"\[[\s\S]*\]", text)
        if array_match:
            try:
                return json.loads(array_match.group())
            except Exception:
                pass

        # -------------------------------------------------
        # Case 3: Extract JSON object from response
        # -------------------------------------------------
        object_match = re.search(r"\{[\s\S]*\}", text)
        if object_match:
            try:
                return json.loads(object_match.group())
            except Exception:
                pass

        # -------------------------------------------------
        # Parsing failed
        # -------------------------------------------------
        print("\nFailed to parse model response.\n")

        return []

    def extract(self, ocr_words, filename=""):
        document = self.builder.build(ocr_words)
        prompt = document.to_prompt(include_words=False)
        prediction = self.module(document=prompt)
        print(prediction.response)
        print()
    
        return {
            "prompt": prompt,
            "raw": prediction.response,
            "parsed": self.parse_json(prediction.response)
        }