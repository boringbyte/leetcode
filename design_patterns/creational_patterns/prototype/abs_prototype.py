from abc import ABC, abstractmethod


class AbstractPrototype(ABC):

    @abstractmethod
    def clone(self):
        pass
