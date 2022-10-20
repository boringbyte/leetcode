from abc import ABC, abstractmethod


class AbstractTree(ABC):

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def accept(self, visitor):
        pass
