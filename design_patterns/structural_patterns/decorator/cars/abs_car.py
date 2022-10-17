from abc import ABC, abstractmethod


class AbstractCar(ABC):

    @property
    @abstractmethod
    def description(self):
        pass

    @property
    @abstractmethod
    def cost(self):
        pass
