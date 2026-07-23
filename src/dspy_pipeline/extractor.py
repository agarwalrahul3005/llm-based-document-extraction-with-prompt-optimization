import json
import re
import dspy

from src.layout.raw_document_builder import RawDocumentBuilder
from layout.normalized_document_builder import NormalizedDocumentBuilder
from dspy_pipeline.module import FormExtractionModule
from prompting.prompt_factory import PromptFactory


class DSPyExtractor:

    def __init__(self, prompt_name="baseline"):

        self.builder = RawDocumentBuilder()
        # self.builder = NormalizedDocumentBuilder()
        self.prompt_builder = PromptFactory.create(prompt_name)


        self.module = FormExtractionModule()

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
        array_match = re.search(
            r"\[[\s\S]*\]",
            text
        )

        if array_match:

            try:
                return json.loads(
                    array_match.group()
                )
            except Exception:
                pass

        # -------------------------------------------------
        # Case 3: Extract JSON object from response
        # -------------------------------------------------
        object_match = re.search(
            r"\{[\s\S]*\}",
            text
        )

        if object_match:

            try:
                return json.loads(
                    object_match.group()
                )
            except Exception:
                pass

        # -------------------------------------------------
        # Parsing failed
        # -------------------------------------------------
        print("\nFailed to parse model response.\n")

        return []

    def extract(self, ocr_words, filename=""):

        # -------------------------------------------------
        # Build structured document
        # -------------------------------------------------
        document = self.builder.build(
            ocr_words
        )

        # -------------------------------------------------
        # Build prompt
        # -------------------------------------------------
        prompt = self.prompt_builder.build(
            document
        )

        # -------------------------------------------------
        # Call DSPy module
        # -------------------------------------------------

        prediction = self.module(document=prompt)


        # -------------------------------------------------
        # Debug output
        # -------------------------------------------------
        print("\n" + "=" * 100)
        print(f"FILE : {filename}")
        print("=" * 100)
        print(prediction.response)
        print("=" * 100 + "\n")
        # print(dspy.inspect_history())
        # print("=" * 100 + "\n")

        # -------------------------------------------------
        # Parse model response
        # -------------------------------------------------
        
        # parsed_prediction = self.parse_json(prediction.response)
        # return parsed_prediction
    
        return {
            "prompt": prompt,
            "raw": prediction.response,
            "parsed": self.parse_json(prediction.response)
        }