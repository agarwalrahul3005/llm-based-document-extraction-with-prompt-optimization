from abc import ABC, abstractmethod


class BasePrompt:

    @abstractmethod
    def build(self, document):
        pass