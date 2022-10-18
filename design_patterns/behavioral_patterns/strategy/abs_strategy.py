from abc import ABC, abstractmethod


class AbstractStrategy(ABC):

    @abstractmethod
    def calculate(self, order):
        """Calculate shipping cost"""
        pass
