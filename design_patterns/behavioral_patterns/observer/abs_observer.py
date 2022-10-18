from abc import ABC, abstractmethod


class AbstractObserver(ABC):

    @abstractmethod
    def update(self, value):
        pass
