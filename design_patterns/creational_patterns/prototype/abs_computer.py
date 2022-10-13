from abc import ABC, abstractmethod


class AbstractComputer(ABC):

    @abstractmethod
    def display(self):
        pass
