import json
import dspy

from dspy_pipeline.signature import ExtractForm


class FormExtractionModule(dspy.Module):

    def __init__(self):
        super().__init__()
        # self.extract = dspy.Predict(ExtractForm)
        self.extract = dspy.ChainOfThought(ExtractForm)

    # def parse_response(self, response):
    #     try:
    #         return json.loads(response)
    #     except Exception:
    #         return []    

    def forward(self, document):
        prediction = self.extract(document=document)
        return prediction