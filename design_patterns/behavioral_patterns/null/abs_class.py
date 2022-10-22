from abc import ABC, abstractmethod


class AbstractClass(ABC):

    @abstractmethod
    def do_something(self, value):
        pass
