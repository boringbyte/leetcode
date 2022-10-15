from abc import ABC, abstractmethod


class AbstractComposite(ABC):

    @abstractmethod
    def get_oldest(self):
        pass
