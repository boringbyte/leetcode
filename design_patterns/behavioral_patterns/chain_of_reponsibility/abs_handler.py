from abc import ABC, abstractmethod


class AbstractHandler(ABC):

    @property
    @abstractmethod
    def successor(self):
        pass

    @property
    @abstractmethod
    def handle(self, request):
        pass
