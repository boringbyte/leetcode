from abc import ABC, abstractmethod


class AbstractAuto(ABC):

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def stop(self):
        pass
