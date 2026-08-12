import json
import dspy

from dspy_pipeline.signature import ExtractForm


class FormExtractionModule(dspy.Module):

    def __init__(self):
        super().__init__()
        # self.extract = dspy.Predict(ExtractForm)
        self.extract = dspy.ChainOfThought(ExtractForm)  

    def forward(self, document):
        prediction = self.extract(document=document)
        return prediction