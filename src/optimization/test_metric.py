import json

from optimization.metric import extraction_metric


class DummyPrediction:
    def __init__(self, response):
        self.response = response


class DummyExample:
    def __init__(self, response):
        self.response = response


gt = [
    {
        "question": "Name",
        "answer": "Rahul"
    }
]

pred = [
    {
        "question": "Name",
        "answer": "Sourav"
    }
]

example = DummyExample(json.dumps(gt))

prediction = DummyPrediction(json.dumps(pred))

print(extraction_metric( example,prediction))