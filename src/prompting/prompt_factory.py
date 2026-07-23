from prompting.baseline_prompt import BaselinePrompt
from prompting.prompt_v2 import PromptV2
from prompting.normalized_prompt import NormalizedPrompt

class PromptFactory:

    @staticmethod
    def create(name):

        prompts = {
            "baseline": BaselinePrompt,
            "v2": PromptV2,
            "normalized": NormalizedPrompt
        }

        return prompts[name]()
    