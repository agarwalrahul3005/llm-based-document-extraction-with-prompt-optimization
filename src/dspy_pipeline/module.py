import dspy

from dspy_pipeline.signature import ExtractForm


class FormExtractionModule(dspy.Module):

    def __init__(self):

        super().__init__()

        self.extract = dspy.Predict(
            ExtractForm
        )

    def forward(self, document):

        return self.extract(
            document=document
        )