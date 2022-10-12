from abc import ABC, abstractmethod


class AbstractFactory(ABC):

    @abstractmethod
    def create_auto(self):
        pass
