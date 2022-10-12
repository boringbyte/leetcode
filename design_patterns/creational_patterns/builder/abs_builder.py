from abc import ABC, abstractmethod
from my_computer import Computer


class AbstractBuilder(ABC):

    def get_computer(self):
        return self._computer

    def new_computer(self):
        self._computer = Computer()

    @abstractmethod
    def get_case(self):
        pass

    @abstractmethod
    def build_mainboard(self):
        pass

    @abstractmethod
    def install_mainboard(self):
        pass

    @abstractmethod
    def install_video_card(self):
        pass

    @abstractmethod
    def install_hard_drive(self):
        pass
