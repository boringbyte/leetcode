from abc import ABC, abstractmethod


class AbstractExpression(ABC):

    @abstractmethod
    def interpret(context):
        pass
